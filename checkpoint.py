import torch
import os
from src.utils import params_to_file


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

    def save(self, path):
        checkpoint_dir = os.path.join(path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{self.global_step}.pth")
        torch.save(self, checkpoint_path)
        params_path = os.path.join(path, "params.txt")
        if not os.path.isfile(params_path):
            params_to_file(self.params, params_path)

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
        from src.hvae import hVAE

        self.global_step = state["global_step"]
        self.model =       hVAE.deserialize(state["model"])
        self.scheduler_state_dict = state["scheduler_state_dict"]
        self.optimizer_state_dict = state["optimizer_state_dict"]


