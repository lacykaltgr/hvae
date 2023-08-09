import torch
import os

class Experiment:
    def __init__(self, global_step=-1, model=None, optimizer=None, scheduler=None, params=None):
        try:
            self.global_step: int = global_step
            self.model = model
            self.model_params = params.model_params
            self.data_params = params.data_params
            self.train_params = params.train_params
            self.optimizer_params = params.optimizer_params
            self.loss_params = params.loss_params
            self.eval_params = params.eval_params
            self.synthesis_params = params.synthesis_params

            self.scheduler_state = scheduler.state_dict()
            self.optimizer_state = optimizer.state_dict()
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
        return torch.load(path)

    @staticmethod
    def load_model(path):
        return torch.load(path)

    def __getstate__(self):
        return {"global_step": self.global_step,
                "model": self.model,
                "model_params": self.model_params,
                "data_params": self.data_params,
                "train_params": self.train_params,
                "optimizer_params": self.optimizer_params,
                "loss_params": self.loss_params,
                "eval_params": self.eval_params,
                "synthesis_params": self.synthesis_params,
                "scheduler_state": self.scheduler_state,
                "optimizer_state": self.optimizer_state}

    def __setstate__(self, state):
        self.global_step = state["global_step"]
        self.model = state["model"]
        self.model_params = state["model_params"]
        self.data_params = state["data_params"]
        self.train_params = state["train_params"]
        self.optimizer_params = state["optimizer_params"]
        self.loss_params = state["loss_params"]
        self.eval_params = state["eval_params"]
        self.synthesis_params = state["synthesis_params"]
        self.scheduler_state = state["scheduler_state"]
        self.optimizer_state = state["optimizer_state"]


