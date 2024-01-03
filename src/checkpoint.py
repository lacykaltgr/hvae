import torch
import os
from src.utils import wandb_log_checkpoint
import wandb


class Checkpoint:
    """
    Checkpoint class for saving and loading experiments
    """
    def __init__(self, global_step=-1, model=None, optimizer=None, scheduler=None, params=None):
        try:
            self.global_step: int = global_step
            self.model = model
            self.params = params

            self.scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
            self.optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        except TypeError:
            print("Error loading experiment")

    def save(self, path, run=None):
        checkpoint_dir = os.path.join(path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self, checkpoint_path)

        if run is not None:
            wandb_log_checkpoint(run, checkpoint_path, self.params.log_params.name)
        return checkpoint_path

    def save_migration(self, path):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"migrated_checkpoint.pth")
        torch.save(self, checkpoint_path)
        return checkpoint_path

    @staticmethod
    def load(path):
        experiment: Checkpoint = torch.load(path)
        return experiment

    def get_model(self):
        return self.model

    def __getstate__(self):
        return {
                "global_step": self.global_step,
                "model":       self.model.serialize(),
                "scheduler_state_dict": self.scheduler_state_dict,
                "optimizer_state_dict": self.optimizer_state_dict
                }

    def __setstate__(self, state):
        from src.hvae.hvae import hVAE

        self.global_step = state["global_step"]
        self.model =       hVAE.deserialize(state["model"])
        self.scheduler_state_dict = state["scheduler_state_dict"]
        self.optimizer_state_dict = state["optimizer_state_dict"]


