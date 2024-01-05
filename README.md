# hVAE: Customizable implementation of Hierarchical VAEs

## Installation
This repository was designed to be used as a template starter project for implementing custom hVAE models in PyTorch with minimal amount of coding.
For installation, use it as a template for creating your custom hVAE repository. (*GitHub > Use this template*)

**Configure docker container (optional)**

```bash
cd path_to_repo/hvae
cd config_docker
docker compose up
```
A JupyterLab server will be created automatically. 
You can access this at **localhost:8880**.

**Weight and Biases**

This repository uses [Weights and Biases](https://wandb.ai/site) for logging and visualization.
To use this, you need to create a free account on their website and login to the CLI.
```bash
wandb login <your_api_key>
```

## Project Structure

```
├── config_docker                   # Docker configuration files
├── data                            # Datasets, dataset loaders
├── eval_notebooks                  # Jupyter notebooks for evaluation
├── experiments                     # Model checkpoints, analysis results
├── migration                       # Migration scripts
├── models                          # Model definitions
├── scripts                         # Python scripts to create files
│   ├── templates                   # Model, dataset templates
├── src
│   ├── elements
│   │   ├── data_preproc.py         # Modules for data preprocessing
│   │   ├── dataset.py              # Base dataset class
│   │   ├── distributions.py        # Distributions, distributions generation
│   │   ├── layers.py               # Layers for building models
│   │   ├── losses.py               # Loss functions
│   │   ├── nets.py                 # Network architectures
│   │   ├── optimizers.py           # Optimizers
│   │   ├── schedules.py            # Schedules e.g. LR, KL weight
│   ├── hvae
│   │   ├── analysis_tools.py       # Analysis tools for trained models
│   │   ├── block.py                # Blocks for building hierarchical models
│   │   ├── hvae.py                 # Universal hVAE class
│   │   ├── model.py                # hVAE model functions (training, loss, etc.)
│   ├── analyze.py                  # Analysis script
│   ├── checkpoint.py               # Checkpoint handling (save, load)
│   ├── hparams.py                  # Model chooser!, hyperparameters class
│   ├── migrate.py                  # Migration script
│   ├── test.py                     # Test script
│   ├── train.py                    # Training script
│   ├── utils.py                    # Utility functions
```

## Create your model

The main aim of this framework is to make defining hierarchical VAE models as simple as possible.
For this purpose, any model can be defined in a single file. 
This file will contain every information about the model, including the model architecture, loss function, optimizer, etc.
To create your model, you need to create a new file in the `models` directory. This can be done two ways:
1. Copy the `scripts/templates/model_template.py` file and rename it to your desired model name.
2. Use the `create_model.py` script in the `scripts` directory. 
    ```bash
    python scripts/create.py model <model_name>
    ```
    This script will create a new model file with the given name.

The model file has the following structure:
```python
def _model():
    from src.hvae.hvae import hVAE as hvae
    from src.utils import OrderedModuleDict
    # block imports
    
    _blocks = OrderedModuleDict(
       # block definitions
    )
    __model = hvae(
        blocks=_blocks,
    )
    return __model


from src.hparams import Hyperparams

# general hyperparameters

log_params = Hyperparams(/* log params */)
model_params = Hyperparams(/* model params */)
data_params = Hyperparams(/* data params */)
train_params = Hyperparams(/* train params */)
optimizer_params = Hyperparams(/* optimizer params */)
loss_params = Hyperparams(/* loss params */)
eval_params = Hyperparams(/* eval params */)
analysis_params = Hyperparams(/* analysis params */)

# default network hyperparameters

mlp_params = Hyperparams(/* mlp params */)
cnn_params = Hyperparams(/* cnn params */)
pool_params = Hyperparams(/* pool params */)
unpool_params = Hyperparams(/* unpool params */)

# custom network hyperparameters
your_custom_net_params = Hyperparams(/* your custom net params */)
```
Start by giving your model a name in the `name` field of `log_params`. 
When you run your model, the results will be saved in the `experiments` directory under this name.

Now you can start building your model architecture by defining its blocks.

### Blocks

Blocks are the building blocks of hierarchical VAEs. Every block has a **name**, **inputs** and a **network**.  
You can define your blocks in the `_blocks` dictionary in the `_model()` function. 
The keys of the dictionary are the names of the blocks, and the values are the blocks themselves.
Every block must be inherited from the base `_Block` class in `src.hvae.block`.

- The first block in the dictionary must be an `InputBlock`. This block initializes the computation graph.
In the `net` parameter of the `InputBlock` preprocessing transformations (e.g. flattening) can be applied.
- The seperation between the Encoder and the Generator networks of the model is automatically detected at the first *GenBlock*.
One of these blocks must be defined in the dictionary.
- The last block in the dictionary must be an `OutputBlock`. 

To describe the connections between blocks just use the name of the blocks as inputs in the `input_id` parameter of the block.
The same applies to the `condition` parameter in generative blocks.  
*TODO: inputpipeline*  
The `net` parameter of the blocks describes a transformation that is applied to the input. 
You can pass any transformation here just make sure it is wrapped by the `SerializableModule` class for serialization.
For ease of use, some common neural networks are already implemented in `src.hvae.nets`. 
These can also be defined with configuration dictionaries (check out `mlp_params` and `cnn_params` in your model file). 

<details>
  <summary>More information on available blocks</summary>

**InputBlock**  
-  Initializes the computation graph. Must be the first block in the dictionary.
- `net`: Preprocessing transformation applied to the input.

**OutputBlock**
-  Final block of the model. Must be the last block in the dictionary.
-  Same as the `SimpleGenBlock`.
- `net`: Transformation applied to the input.
- `input_id`: Name of the input block.
- `output_distribution`: Type of the returned distribution.

**SimpleBlock**
- Applies a transformation on its input.
- `net`: Transformation applied to the input.
- `input_id`: Name of the input block.

**DualInputBlock**
- Applies a transformation on its input.
- `net`: Transformation applied to the inputs.
- `input_ids`: Names of the 2 input blocks.

**TopSimpleBlock**
- Bottleneck block, has a (trainable) prior value. Applies a transformation on the prior.
- `net`: Transformation applied to the prior.
- `prior_shape`: Shape of the prior.
- `prior_trainable`: Whether the prior is trainable.
- `prior_data`: Initial value of the prior.

**SimpleGenBlock**
- Takes an input, applies a transformation and samples from a prior distribution 
parameterized by the transformed input.
- `net`: Transformation applied to the input.
- `output_distribution`: Type of the prior distribution.
- `input_id`: Name of the input block.

**GenBlock**
- Takes an input, applies a transformation, samples from a prior distribution given by the transformed input. 
- Takes a condition, (concatenates it with the input), samples from a posterior distribution given by the transformation.
- `prior_net`: Transformation applied to the input.
- `posterior_net`: Transformation applied to the condition (concatenated with the input).
- `input_id`: Name of the input block.
- `condition`: Name of the condition block.
- `concat_posterior`: Whether the condition is concatenated with the input before applying the posterior transformation.
- `output_distribution`: Type of the prior and posterior distributions.
- `input_transform`: Transformation applied to the input before concatenating with the condition.
- `condition_transform`: Transformation applied to the condition before concatenating with the input.

**TopGenBlock**
- Bottleneck block, has a (trainable) prior value. Applies a transformation on the prior.
- Takes a condition, (concatenates it with the input), samples from a posterior distribution given by the transformation.
- `net`: Transformation applied to the condition (concatenated with the prior).
- `prior_shape`: Shape of the prior.
- `prior_trainable`: Whether the prior is trainable.
- `prior_data`: Initial value of the prior.
- `condition`: Name of the condition block.
- `concat_posterior`: Whether the condition is concatenated with the prior before applying the posterior transformation.
- `output_distribution`: Type of the prior and posterior distributions.

**ResidualGenBlock**
- Architecture from VDVAE paper 
- [https://arxiv.org/abs/2011.10650](https://arxiv.org/abs/2011.10650)


**Custom blocks**  

You can design your own blocks by inheriting from the `_Block` class 
or by inheriting from pre-defined blocks and overriding just the specific functionalities.  
Any custom block must realize the `serialize()` and the `deserialize(serialized)` methods for handling serialization.  
Any custom *Encoder* block must realize this method:
- `forward(computed)`: Forward pass of the block. Returns the updated `computed` dictionary.

Any custom *Generator* block must realize these methods:
- `forward(computed, use_mean)`: Forward pass of the block. Returns the updated `computed` dictionary along with a tuple containing the prior and posterior distributions.
- `sample_from_prior(computed, use_mean=False)`: Samples from the prior distribution. Returns the updated `computed` dictionary along with the sampled prior.

</details>

When you **run the model**, the computation graph will be automatically built from the blocks.
For given inputs, the model will return with a tuple of 2 dictionaries, both indexed by the names of the blocks
   - `computed` contains the (sampled) outputs of the blocks.
   - `distributions` contains in tuples the prior (and posterior) distributions of the generative blocks.



## Datasets

To get your dataset ready for training, you need to create a new dataset class in the `data` directory.
Similar to the model file, you can use the `create_dataset.py` script in the `scripts` directory to create a new dataset file or copy the `scripts/templates/dataset_template.py` file and rename it to your desired dataset name.

The dataset file has the following structure:
```python
from src.elements.dataset import _DataSet

class YourDataset(_DataSet):
    def __init__(self, with_labels=False):
        super(YourDataset, self).__init__(with_labels=with_labels)

    def load(self):
        x_train, y_train, x_val, y_val, x_test, y_test = self._your_dataset_loader()
        
        if self.with_labels:
            return x_train, y_train, x_val, y_val, x_test, y_test
        else:
            return x_train, x_val, x_test
```

The `load()` method must return the dataset in the following format:
- If the dataset has labels (initialization parameter `with_labels=True`)):
    - `x_train, y_train, x_val, y_val, x_test, y_test`
- If the dataset has no labels (initialization parameter `with_labels=False`)
    - `x_train, x_val, x_test`
  
The framework will automatically wrap these into PyTorch `DataLoader` objects 
and provide you with a simple interface for easy access.  

To **run the model**, on your dataset you need to specify 
the dataset in the `data_params` hyperparameters object in your model file.
```python
from data.YourDataSet import YourDataset
data_params = Hyperparams(
    dataset=YourDataset,
    params=dict(with_labels=False),
    shape=(1, 28, 28),  # Shape of the your dataset samples
    num_bits=8.,        # Number of bits used to encode each channel 
)
```
If you would like to use a dataset that is already implemented in the framework,
you can find them in the `data` directory. 

On the loaded dataset you can access the following methods:
- `get_train_loader()`: Returns a `DataLoader` object for the training set.
- `get_val_loader()`: Returns a `DataLoader` object for the validation set.
- `get_test_loader()`: Returns a `DataLoader` object for the test set.


## Training

To **train your model**, you need to:

1. Configure the training **hyperparameters** in your model file. See the comments in your model file for more information.
2. Import your model in the `hparams.py` file.
   ```python
   def get_hparams():
      # SET WHICH params TO USE HERE
      # |    |    |    |    |    |
      # v    v    v    v    v    v
      import models.YourModel as params
      ...
   ```
3. Run `train.py` 
   ```bash
   python src/train.py
   ```
   This will start training your model with the given hyperparameters.
   The training set of the dataset will be used for training and the validation set for validation.

**Training logs**  

When you start training, a new directory will be created in the `experiments` directory with the name of your model.
In this directory, another directory will be created timestamped with the current date and time (or other naming set in the hyperparameters).
This directory will contain the following:
- `checkpoints`: Model checkpoints will be saved here. Checkpointing interval can be set in the hyperparameters.
- `tensorboard`: Tensorboard logs will be saved here. Logging interval can be set in the hyperparameters.
- `log.txt`: A text file containing the training logs.
- `params.json`: A json file containing the hyperparameters used for training.
- `train_log.csv`: A csv file containing the training logs.
- `val_log.csv`: A csv file containing the validation logs.

**Training from a checkpoint**  

To train your model from a checkpoint, you need to set the `load_from_train` parameter in the `log_params` of your model file to the path to the checkpoint you want to start from.

## Evaluation

To **evaluate your model**, you need to:

1. Configure the evaluation **hyperparameters** in your model file. See the comments in your model file for more information.
2. Import your model in the `hparams.py` file. This is the same as for training.
3. Run `eval.py` 
   ```bash
   python src/eval.py
   ```
   This will start evaluating your model with the given hyperparameters on the test set of your dataset. 

## Analysis

To **analyze your model**, you need to:
1. Configure the analysis **hyperparameters** in your model file. 
2. Import your model in the `hparams.py` file. This is the same as for training and evaluation.
3. Run `analyze.py` 
   ```bash
   python src/analyze.py
   ```
   This will start analyzing your model with the given hyperparameters and save the results in the `experiments/YourModel/analysis` directory.

The framework provides several analysis tools for trained models.
These include:
- **Reconstruction**: reconstructs samples from the test set and provides a reconstruction statistics.
- **Generation**: generates samples from the prior distribution and provides a generation statistics.
- **Divergence statistics**: computes the KL divergence between the prior and the posterior distributions of the generative blocks.
- **Decodability**: computes the decodability of the latent representations of the test set.
- **White noise analysis**: generates visualizations of the receptive fields of the latent dimensions.
- **Latent step analysis**: generates visualizations of the latent space traversal of the test set.
- **Most exciting input (MEI)**: finds the most exciting input for each latent dimension.

## Migrate trained models

To **migrate your trained model**, you to this framework, first you need to 
1. Create a new model file for it in the `models` directory.  
You can use the `create_migration.py` script in the `scripts` directory to create a new model file or copy the `scripts/templates/migration_template.py` file and rename it to your desired model name.
The migration file follows a very similar structure to the model file with a few differences:
   - The `_model()` function takes an additional `migration` parameter.
      ```python
       def _model(migration):
           ...
       ```
   - A new hyperparameters object `migration_params` is defined in the file.
      ```python
      from migration.your_migration_agent import YourMigrationAgent
  
      migration_params = Hyperparams(
           migration_agent=YourMigrationAgent,
           params=dict(
               # params to migration agent e.g path="<path_to_model_weights>"
           )
      )
2. Implement a migration agent in the `migration` directory.  
You can use the `create_migration_agent.py` script in the `scripts` directory to create a new migration agent file or copy the `scripts/templates/migration_agent_template.py` file and rename it to your desired migration agent name.  
The migration agent:
   - must take the dictionary of parameters defined in the `migration_params` hyperparameters object as argument in its constructor
   - migrates the model weights to the new model
   - provides an interface to access these weights as valid networks in the hVAE framework
3. Use the migration argument in the `_model()` function to load networks with the weights of the old model.
   ```python
   def _model(migration):
       ...
       _blocks = OrderedDict(
        ...
   
        "z" = SimpleBlock(
            net=migration.get_network("z"),
            input_id="x"
        ),
       ...
   ```
4. Run `migrate.py` 
   ```bash
   python src/migrate.py
   ```
   This will start migrating your model with the given hyperparameters and save the results in the `experiments/YourModel/migration` directory.

