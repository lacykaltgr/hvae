def get_hparams():
    # SET WHICH params TO USE HERE
    # |    |    |    |    |    |
    # v    v    v    v    v    v
    import models.SMTVAE as params

    config = Hyperparams(
        log_params=params.log_params,
        model_params=params.model_params,
        data_params=params.data_params,
        train_params=params.train_params,
        optimizer_params=params.optimizer_params,
        loss_params=params.loss_params,
        eval_params=params.eval_params,
        analysis_params=params.analysis_params,
    )

    if hasattr(params, 'migration_params'):
        config.migration_params = params.migration_params

    return config


class Hyperparams:
    def __init__(self, **config):
        self.config = config

    def __getattr__(self, name):
        if name == 'config':
            return super().__getattribute__(name)
        return self.config[name]

    def __setattr__(self, name, value):
        if name == 'config':
            super().__setattr__(name, value)
        else:
            self.config[name] = value

    def __getstate__(self):
        return self.config

    def __setstate__(self, state):
        self.config = state

    def keys(self):
        return self.config.keys()

    def __getitem__(self, item):
        return self.config[item]

    def to_json(self):
        from types import FunctionType
        from elements.dataset import _DataSet

        def convert_to_json_serializable(obj):
            if isinstance(obj, Hyperparams):
                return convert_to_json_serializable(obj.config)
            if isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            if callable(obj) or isinstance(obj, FunctionType):
                return str(obj)
            if isinstance(obj, _DataSet):
                return str(obj)
            return obj

        json_serializable_config = convert_to_json_serializable(self.config)
        return json_serializable_config

    @classmethod
    def from_json(cls, json_str):
        import json
        data = json.loads(json_str)
        return cls(**data)
