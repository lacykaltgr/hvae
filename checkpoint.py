import torch
import os


class Checkpoint:
    """
    Checkpoint class for saving and loading experiments
    """
    def __init__(self, global_step=-1, model=None, optimizer=None, scheduler=None, params=None):
        try:
            self.global_step: int = global_step
            self.model = model
            if params is not None:
                self.model_params = params.model_params
                self.data_params = params.data_params
                self.train_params = params.train_params
                self.optimizer_params = params.optimizer_params
                self.loss_params = params.loss_params
                self.eval_params = params.eval_params
                self.synthesis_params = params.synthesis_params

            self.scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
            self.optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        except TypeError:
            print("Error loading experiment")

    def save(self, path):
        if not os.path.isfile(path):
            checkpoint_dir = os.path.join(path, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"checkpoint-{self.global_step}.pth")
        torch.save(self, path)
        return path

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
        self.scheduler_state = state["scheduler_state_dict"]
        self.optimizer_state = state["optimizer_state_dict"]


