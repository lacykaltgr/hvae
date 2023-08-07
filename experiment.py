import torch

class Experiment:
    def __init__(self, global_step=-1, model=None, model_params=None,
                 data_params=None, train_params=None, eval_params=None, synthesis_params=None,
                 scheduler_state=None, optimizer_state=None, optimizer_params=None, loss_params=None):
        try:
            self.global_step: int = global_step
            self.model = model
            self.model_params = model_params
            self.data_params = data_params
            self.train_params = train_params
            self.optimizer_params = optimizer_params
            self.loss_params = loss_params
            self.eval_params = eval_params
            self.synthesis_params = synthesis_params

            self.scheduler_state = scheduler_state
            self.optimizer_state = optimizer_state
        except TypeError:
            print("Error loading experiment")

    def save(self, path):
        torch.save(self, path)

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


