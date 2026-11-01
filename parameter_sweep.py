# Run this in a separate script or notebook:
from TOK_train import main_sweep, wandb
if __name__ == "__main__":
    sweep_id = '7o0wz306'
    wandb.agent(sweep_id, function=main_sweep, count=50, project="tok_hydrology_sweep")
