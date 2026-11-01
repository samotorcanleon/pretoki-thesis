import os
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm
import gin
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import plotting_and_evaluation_functions as pef

from TOK_dataset import TOKDataset, create_tok_dataloaders, create_cv_dataloaders
# Assuming you have a TOK model defined somewhere
from TOK.TOK import TOK

class TOKTrainer:
    def __init__(self, config):
        """
        Initialize the TOK trainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Set seed for reproducibility
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        
        # Create model
        self.model = TOK().to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_tok_dataloaders(
            station_id=config['station_id'],
            batch_size=config['batch_size'],
            train_up_to_year=config['train_up_to_year'],
            meteo_past_hours=config['meteo_past_hours'],
            prediction_hours=config['prediction_hours'],
            river_history_hours=config['river_history_hours'],
            num_closest_meteo_stations=config['num_closest_meteo_stations']
        )
        output_scaler = train_loader.dataset.scalers['Q'] # sklearn scaler
        TOK.output_scaler = output_scaler
        
        self.dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }


        # Create scheduler
        
        # CosineAnnealingLR scheduler: learning rate decreases following a cosine curve
        # from initial lr to eta_min over T_max epochs, then restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],     # Total number of epochs
            eta_min=config['lr']*config['min_lr_factor']                # Minimum learning rate
        )
        # Loss function - asymmetric loss that penalizes underprediction more than overprediction
        # Using a custom loss function where alpha controls the asymmetry
        # alpha > 1 means underprediction is penalized more
        alpha = 1.5  # Adjust this value to control how much more underprediction is penalized
        self.criterion = lambda pred, target: torch.mean(
            torch.where(pred < target, 
                        alpha * (target - pred) ** 2,  # Underprediction case
                        (pred - target) ** 2)  # Overprediction case
        )

        
        # Create directory for saving models
        os.makedirs(config['model_dir'], exist_ok=True)
    
    def train_epoch(self):
        """Train model for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in self.dataloaders['train']:
            # Move data to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
                
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Get target and mask NaN values
            targets = batch['target']

            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping
            if self.config['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
                
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / max(1, num_batches)
    
    def evaluate(self, phase):
        """Evaluate model on validation or test set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.dataloaders[phase]:
                # Move data to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Get target and mask NaN values
                targets = batch['target']
                mask = ~torch.isnan(targets)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for additional metrics
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Compute additional metrics if needed
        
        return total_loss / max(1, num_batches)
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        # Create progress bar for epochs
        progress_bar = tqdm(range(self.config['epochs']), desc="Training", leave=True, mininterval=1)
        
        for epoch in progress_bar:
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Evaluate on validation set
            val_loss = self.evaluate('val')
            
            # Evaluate on test set every epoch
            test_loss = self.evaluate('test')
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch
            }
            
            wandb.log(metrics)
            
            # Save model if it has the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(self.config['model_dir'], f"best_model_{wandb.run.id}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'test_loss': test_loss,
                }, model_path)
                #wandb.save(model_path)
            
            # Update progress bar with metrics (refreshes the same line)
            progress_bar.set_postfix({
                'train_loss': f"{train_loss:.6f}",
                'val_loss': f"{val_loss:.6f}",
                'test_loss': f"{test_loss:.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.7f}",
                'best_val': f"{best_val_loss:.6f}"
            })
            progress_bar.update(0)  # Force refresh without advancing
        
        # Save final model
        model_path = os.path.join(self.config['model_dir'], f"final_model_{wandb.run.id}.pth")
        torch.save({
            'epoch': self.config['epochs'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'test_loss': test_loss,
        }, model_path)
        #wandb.save(model_path)

        predictions, targets, prediction_timestamps = self.scaled_predictions_with_best_model(self.dataloaders['test'])

        metrics_df, fig_predictions_analysis, fig_metrics, fig_biggest_miss, _ = pef.plot_common(self.config['station_id'], targets, predictions, prediction_timestamps)
        wandb.log({
                    "predictions_analysis": wandb.Image(fig_predictions_analysis),
                    "metrics_plot": wandb.Image(fig_metrics),
                    "biggest_miss_plot": wandb.Image(fig_biggest_miss),
                    "metrics_table": wandb.Table(dataframe=metrics_df),
                    "best_val_loss": best_val_loss
                    })



    def scaled_predictions_with_best_model(self, dataloader):
        """
        Make predictions using the best model on the given dataloader.
        
        Args:
            dataloader: DataLoader to make predictions on
            
        Returns:
            tuple: (predictions, targets, prediction_timestamps) - scaled back to original values
        """
        # Load the best model
        best_model_path = os.path.join(self.config['model_dir'], f"best_model_{wandb.run.id}.pth")
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_prediction_timestamps = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch)

                # Scale predictions and targets back to original values
                scaled_preds = self.model.scaled_predictions(predictions)
                scaled_targets = self.model.scaled_predictions(batch['target'])
                
                all_predictions.append(scaled_preds)
                all_targets.append(scaled_targets)
                all_prediction_timestamps.append(batch['prediction_timestamp'].cpu().numpy().flatten())
                
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0).reshape(-1,18)
        all_targets = np.concatenate(all_targets, axis=0).reshape(-1,18)
        all_prediction_timestamps = np.concatenate(all_prediction_timestamps, axis=0).flatten()
        all_prediction_timestamps = list(pd.to_datetime(all_prediction_timestamps, unit='s'))

        return all_predictions, all_targets, all_prediction_timestamps






@gin.configurable
class Config():
    def __init__(self,
                seed=42,
                data_path=None,
                lr=gin.REQUIRED,
                batch_size=gin.REQUIRED,
                epochs=gin.REQUIRED,
                device=gin.REQUIRED,
                station_id=gin.REQUIRED,
                weight_decay=None,
                model_dir=None,
                wandb_project=gin.REQUIRED,
                run_name=None,
                prediction_hours=None,
                river_history_hours=None,
                meteo_past_hours=None,
                num_closest_meteo_stations=None,
                input_dim=None,
                hidden_dim=None,
                clip_grad_norm=None,
                train_up_to_year=None,
                variable_names=None,
                patience=None,
                min_lr_factor=None,
                generate_extra_data_points=None):
        self.seed = seed
        self.patience = patience
        self.data_path = data_path
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.station_id = station_id
        self.weight_decay = weight_decay
        self.model_dir = model_dir
        self.wandb_project = wandb_project
        self.run_name = run_name
        self.prediction_hours = prediction_hours
        self.river_history_hours = river_history_hours
        self.meteo_past_hours = meteo_past_hours
        self.num_closest_meteo_stations = num_closest_meteo_stations
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.clip_grad_norm = clip_grad_norm
        self.train_up_to_year = train_up_to_year
        self.variable_names = variable_names
        self.min_lr_factor = min_lr_factor
        self.generate_extra_data_points = generate_extra_data_points

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

def load_config(config_path, fix_parameters = {}):
    """Load configuration from a file"""
    import gin
    
    # Parse the config file
    gin.parse_config_file(config_path)

    for key, value in fix_parameters.items():
        gin.bind_parameter(key, value)
    
    # Create a config object using gin-configured values
    config = Config()
    
    # Convert to dictionary for compatibility with existing code
    return config.to_dict()

def load_config_sweep(config_path):
    """Load configuration from a file"""
    import gin
    
    # Parse the config file
    gin.parse_config_file(config_path)
    
    
    # Update config with any changes from wandb sweep
    for key, value in wandb.config.items():
        gin.bind_parameter(key, value)
    
    # Create a config object using gin-configured values
    config = Config()
    
    # Convert to dictionary for compatibility with existing code
    return config.to_dict()

def main(trainer_class, run_station_list = None):
    print(f"station list: {run_station_list}")
    """Main function to run training"""
    # Load configuration from file
    import sys
    if len(sys.argv) < 2:
        print("Usage: python TOK_train.py <config_file> [station_id ... | all]")
        sys.exit(1)
    
    config_path = sys.argv[1]

    if run_station_list is None:
        run_parameter_fixes = [{}]
    else:
        run_parameter_fixes = [{'Config.station_id': station_id} for station_id in run_station_list]

    failed_list =[]
    for parameter_fix in run_parameter_fixes:
        try:
            config = load_config(config_path, parameter_fix)
            print(config_path)
            print(config)
            print(config['station_id'])
            
            # Check for CUDA availability
            if config['device'] == 'cuda' and not torch.cuda.is_available():
                print("CUDA not available, using CPU instead.")
                config['device'] = 'cpu'
            
            # Initialize wandb
            if config['run_name'] is None:
                run_name = f"TOK_station_{config['station_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                run_name = config['run_name']

            wandb.init(
                project=config['wandb_project'],
                name=run_name,
                config=config
            )

            # Create trainer and train model
            trainer = trainer_class(config)
            print(f"Starting training with configuration: {config}")
            trainer.train()
            
            # Finish wandb run
            wandb.finish()
        except Exception as e:
            print(f"Error: {e}")
            wandb.finish(exit_code=1)
            failed_list.append((config['station_id'], f"Error: {e}"))

    if failed_list:
        print("Failed stations:")
        for station_id, error in failed_list:
            print(f"Station {station_id}: {error}")

def create_sweep_configuration():
    """Create a wandb sweep configuration based on TOK_config.gin"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'Config.lr': {
                'min': 0.00005,
                'max': 0.0005,
                'distribution': 'log_uniform_values'
            },
            'Config.weight_decay': {
                'min': 0.000001,
                'max': 0.0001,
                'distribution': 'log_uniform_values'
            },
            'Config.batch_size': {
                'values': [32, 64, 128, 256]
            },
            'Config.epochs': {
                'values': [100, 200, 300]
            },
            'Config.min_lr_factor': {
                'min': 0.1,
                'max': 0.7
            },
            'Config.river_history_hours': {
                'values': [4, 12, 24]
            },
            'Config.num_closest_meteo_stations': {
                'values': [1, 3, 5]
            },
            'Config.clip_grad_norm': {
                'values': [0.5, 1.0, 2.0]
            },
            'Config.seed': {
                'value': 42  # Fixed for reproducibility
            },
            'TOK.hidden_dim': {
                'values': [64, 128, 256]
            },
            'TOK.prediction_layer_hidden_dim': {
                'values': [512, 1024, 2048]
            },
            'TOK.dropout_rate': {
                'min': 0.1,
                'max': 0.5
            },
            'SimpleEncoder.hidden_dim': {
                'values': [128, 256, 512]
            },
            'SimpleEncoder.output_dim': {
                'values': [128, 256, 512]
            },
            'SimpleEncoder.dropout_rate': {
                'min': 0.1,
                'max': 0.5
            }
        }
    }
    return sweep_config


def main_sweep():
    """Main function to run training"""
    # Load configuration from file
    with wandb.init(project="tok_hydrology_sweep") as run:
        config_path = 'TOK_config.gin' # hardcoded for now, gets run over by wandb sweep
        config = load_config_sweep(config_path)
        print(config)
        
        # Check for CUDA availability
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead.")
            config['device'] = 'cpu'
        
        # No need to initialize wandb again as it's already initialized in the sweep agent
        # Just use the current run
        
        # Create trainer and train model
        trainer = TOKTrainer(config)
        print(f"Starting training with configuration: {config}")
        trainer.train()


# Stations in the final experiment. Left out because of broken data or missing
# ARSO predictions: 1070, 2110, 2220, 3060, 3280, 3350, 3400, 3570, 5880
FINAL_STATION_LIST = [
    1060,1100,1140,1165,1220,1260,1300,1312,1335,1355,
    2005,2010,2138,2140,2150,2160,2250,2372,2390,2420,2428,2432,2530,2600,2620,2640,2652,2667,2693,2719,2754,2830,2880,2900,
    3014,3080,3105,3108,3115,3180,3200,3250,3260,3300,3320,3420,3465,3530,3550,3660,3725,3850,3900,
    4025,4050,4095,4120,4155,4200,4209,4222,4230,4270,4298,4400,4430,4480,4515,4520,4570,4575,4626,4650,4671,4695,4706,4710,4750,4770,4791,4820,4828,4860,4960,4965,4969,4986,
    5030,5078,5240,5270,5330,5425,5440,5479,5500,5540,5580,5770,5800,5840,5910,5940,
    6020,6060,6068,6120,6140,6200,6210,6220,6240,6280,6300,6340,6415,6550,6630,6691,6720,6770,6790,6835,
    7029,7060,7110,7160,7200,7220,7230,7235,7245,7272,7308,7340,7350,7380,7409,7440,7488,
    8031,8060,8080,8180,8270,8332,8350,8351,8450,8454,8478,8500,8545,8561,8565,8591,8601,8610,8630,8640,8660,8670,8680,8690,8700,8710,
    9015,9030,9050,9077,9100,9108,9210,9240,9275,9280,9300,
]


if __name__ == "__main__":
    # python TOK_train.py <config_file>                  -> station from Config.station_id
    # python TOK_train.py <config_file> 4222 4230        -> the given stations
    # python TOK_train.py <config_file> all              -> FINAL_STATION_LIST
    import sys
    from TOK_cv_train import TOK_CV_Trainer

    station_args = sys.argv[2:]
    if station_args == ["all"]:
        run_station_list = FINAL_STATION_LIST
    elif station_args:
        run_station_list = [int(s) for s in station_args]
    else:
        run_station_list = None
    main(TOK_CV_Trainer, run_station_list=run_station_list)

    # To run a W&B hyperparameter sweep instead (agents can then join with parameter_sweep.py):
    #sweep_config = create_sweep_configuration()
    #sweep_id = wandb.sweep(sweep_config, project="tok_hydrology_sweep")
    #print(f"Sweep ID: {sweep_id}")
    #wandb.agent(sweep_id, function=main_sweep, count=50)