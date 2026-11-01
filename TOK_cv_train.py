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


class TOK_CV_Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        self.cv_splits = create_cv_dataloaders(station_id=config['station_id'],
                                                batch_size=config['batch_size'],
                                                train_up_to_year=config['train_up_to_year'],
                                                meteo_past_hours=config['meteo_past_hours'],
                                                prediction_hours=config['prediction_hours'],
                                                river_history_hours=config['river_history_hours'],
                                                num_closest_meteo_stations=config['num_closest_meteo_stations'],
                                                generate_extra_data_points=config['generate_extra_data_points'])

        # Create directory for saving models
        os.makedirs(config['model_dir'], exist_ok=True)

    def train_epoch(self, dataloader):
        """Train model for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
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
    

    def evaluate(self, dataloader):
        """Evaluate model on  test set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Get target and mask NaN values
                targets = batch['target']
                
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
    

    def train_split(self, train_loader, test_loader):

        # 1. init model in learning related objects

        # Set seed for reproducibility
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])

        # 2. init model in learning related objects
        # Create model
        self.model = TOK().to(self.device)
        # init model prediction scaler
        output_scaler = train_loader.dataset.scalers['Q'] # sklearn scaler
        self.model.output_scaler = output_scaler
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],     # Total number of epochs
            eta_min=self.config['lr']*self.config['min_lr_factor']                # Minimum learning rate
        )   
        # Loss function - asymmetric loss that penalizes underprediction more than overprediction
        alpha = 1.5  # Adjust this value to control how much more underprediction is penalized
        self.criterion = lambda pred, target: torch.mean(
            torch.where(pred < target, 
                        alpha * (target - pred) ** 2,  # Underprediction case
                        (pred - target) ** 2)  # Overprediction case
        )

        
        """Main training loop"""
        # Create progress bar for epochs
        progress_bar = tqdm(range(self.config['epochs']), desc="Training", leave=True, mininterval=1)
        
        for epoch in progress_bar:
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate on test set every epoch
            test_loss = self.evaluate(test_loader)
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch
            }
            
            wandb.log(metrics)
            
            # Update progress bar with metrics (refreshes the same line)
            progress_bar.set_postfix({
                'train_loss': f"{train_loss:.6f}",
                'test_loss': f"{test_loss:.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.7f}"
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

        predictions, targets, prediction_timestamps = self.scaled_predictions_with_final_model(test_loader)
        return predictions, targets, prediction_timestamps

    def scaled_predictions_with_final_model(self, dataloader):
        """
        Make predictions using the best model on the given dataloader.
        
        Args:
            dataloader: DataLoader to make predictions on
            
        Returns:
            tuple: (predictions, targets, prediction_timestamps) - scaled back to original values
        """
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



    def train(self):
        all_predictions = []
        all_targets = []
        all_prediction_timestamps = []
        all_mse_scores = []
        for train_loader, test_loader in self.cv_splits:
            predictions, targets, prediction_timestamps = self.train_split(train_loader, test_loader)
            all_predictions.append(predictions)
            all_targets.append(targets)
            all_prediction_timestamps.append(prediction_timestamps)
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(targets, predictions)
            all_mse_scores.append(mse)

        # Concatenate all predictions, targets, and timestamps from cross-validation splits
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        prediction_timestamps = np.concatenate(all_prediction_timestamps, axis=0)

        # plot and log metrics
        metrics_df, fig_predictions_analysis, fig_metrics, fig_biggest_miss, (arso_predictions_df, model_predictions_df, actuals_df) = pef.plot_common(self.config['station_id'], targets, predictions, prediction_timestamps)
        # some annoying stuff fo wandb.Table
        arso_predictions_df.columns = arso_predictions_df.columns.astype(str)
        model_predictions_df.columns = model_predictions_df.columns.astype(str)
        actuals_df.columns = actuals_df.columns.astype(str)

        wandb.log({
                    "predictions_analysis": wandb.Image(fig_predictions_analysis),
                    "metrics_plot": wandb.Image(fig_metrics),
                    "biggest_miss_plot": wandb.Image(fig_biggest_miss),
                    "metrics_table": wandb.Table(dataframe=metrics_df),
                    "mse_scores": wandb.Table(dataframe=pd.DataFrame(all_mse_scores, columns=['mse_scores'])),
                    "arso_predictions_df": wandb.Table(dataframe=arso_predictions_df),
                    "model_predictions_df": wandb.Table(dataframe=model_predictions_df),
                    "actuals_df": wandb.Table(dataframe=actuals_df)
                    })
















