---
layout: default
title: Equip RL-algorithm with forward model
parent: Wiki
has_children: false
nav_order: 2
---

You can extend any algorithm with a forward model that learns the environments transition dynamics, i.e., that predicts the `next_observation`, given `observation` and `action`.

# Prerequisite

Please ensure that you have completed the tutorial for [adding a new algorithm](Adding-a-new-Algorithm.md) before proceeding with this guide.

# Necessary steps

The following basic steps are necessary to equip an algorithm of your choice with a forward model:

1. Add a new algorithm as described in the corresponding section in this wiki.
2. Extend the new algorithms configuration by a field `fwd`, that contains important parameters for the forward model.
3. Import the desired forward model classes from `src.utils.forward_models.py` and a buffer for the training data from `src.utils.fw_utils.py` (you could also use a replay buffer class provided by stable baselines, but we recommend to use our custom class for convenience).
5. Instanciate the forward model and the data buffer.
6. Find the appropriate places to collect training data and to call your forward models `train` method.
7. (Optional) Save your model for later use

Here, we exemplary show you how to equip the algorithm `cleansac` with a forward model.

## Adding a new algorithm

Start with copying the folder `src.custom_algorithms.cleansac`. Name your copy `cleansac_mod_fw`. 

Copy the file `conf.algorithm.cleansac.yaml`. Name your copy `cleansac_mod_fw.yaml`. 

IMPORTANT: Remember to change the name of your algorithm class, the import in the `__init__.py` and the name in the `cleansac_mod_fw.yaml`. If you do not know what this means, please read the section about adding a new algorithm first!

## Extending the configuration

In your newly created configuration file `cleansac_mod_fw.yaml`, add a field `fwd`, that contains all parameters for your forward model. This looks as follows:

```
# Original file

name: 'cleansac'

learning_rate: 0.0003
buffer_size: 1_000_000
learning_starts: 1000
batch_size: 256
tau: 0.005
gamma: 0.99
ent_coef: auto
use_her: True
n_critics: 2

# Whether to set future expected discounted cumulative reward to what the critic
# computes or to just set it to 0 if the episode is done. The SB3 implementation sets it to 0 if done,
# which is equivalent to false, so we set this as default here.
ignore_dones_for_qvalue: False

# Usually, the action scale is determined by the action space of the environment. Sometimes, however, it is
# desireable to not max out the full scale and reduce the action intensity. For this purpose,
# action_scale_factor is multiplied with the action space of the environment.
action_scale_factor: 1.0

# Whether to log each observation and action dimension in each step. This might slow down everything, but it can help
# debugging because it logs each action and observation dimension per step. It only holds for training, not for eval.
log_obs_step: False
log_act_step: False

```
```
# Changed file

name: 'cleansac_mod_fw'

learning_rate: 0.0003
buffer_size: 1_000_000
learning_starts: 1000
batch_size: 256
tau: 0.005
gamma: 0.99
ent_coef: auto
use_her: True
n_critics: 2

# Whether to set future expected discounted cumulative reward to what the critic
# computes or to just set it to 0 if the episode is done. The SB3 implementation sets it to 0 if done,
# which is equivalent to false, so we set this as default here.
ignore_dones_for_qvalue: False

# Usually, the action scale is determined by the action space of the environment. Sometimes, however, it is
# desireable to not max out the full scale and reduce the action intensity. For this purpose,
# action_scale_factor is multiplied with the action space of the environment.
action_scale_factor: 1.0

# Whether to log each observation and action dimension in each step. This might slow down everything, but it can help
# debugging because it logs each action and observation dimension per step. It only holds for training, not for eval.
log_obs_step: False
log_act_step: False

fwd:
  #  Parameters for the forward model
  hidden_size: 128
  n_hidden_layers : 2
  activation_func : 'relu'  # can be 'relu', 'tanh' or 'logistic'
  loss_func : 'l2'  # for deterministic models, use l2; for probabilisticMLE model use nll. You can implement your desired loss in src.utils.forward_models.py
  fw_batch_size : 256
  fw_learning_rate : 0.0001
  train_every_n_data : 2000  # can be set to 1 for algos that already train only in every k'th step
  ensemble_size : 3
  predict_reward: 1 # can be set to 0 if you don't want to predict the reward, but only the next state
  model_save_path : 'your/path/here'
  pre_train_data_path : 'your/path/here'

```

### Forward model configuration

1. hidden_size: Number of neurons in each hidden layer.
2. n_hidden_layers: Numer of hidden layers.
3. activation_func: Activation function.
4. loss_func: Empirical risk function.
5. fw_batch_size: Batch size for training the forward model.
6. fw_learning_rate: Learning rate for training the forward model.
7. train_every_n_steps: In most cases, it makes sense to retrain the forward model only after n new data points have been collected. Training after every step slows the process down considerably.
8. ensemble_size: Specifies the number of models in the ensemble. Set to `1` for a single model.  
9. predict_reward: Determines whether the model predicts only the next state (`0`) or both the next state and the reward (`1`).  
10. model_save_path: Defines the file path where the model or its `state_dict` should be saved.  
11. pre_train_data_path: Specifies the file path for pretraining data. If using pretraining, provide the path to a `.pt` file. For more details on saving and loading models, refer to the [PyTorch documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

## Import the desired forward model class and the training data buffer

So far, we provide two basic types of fully connected, dense MLP.

The class `DeterministicForwardModel` is a point estimator for the next observation.

The class `ProbabilisticForwardMLENetwork` learns a normal distribution over the predicted next observation.

Import the class you need to your `cleansac_mod_fw.py` as follows:

```
"""
Imports for the fw models
"""
from utils.forward_models import DeterministicForwardNetwork, ProbabilisticForwardMLENetwork
```


## Instanciate forward model and data buffer

First, it is crucial that you add `fwd = {}` as an argument in your algorithms initializer:

```
class CLEANSAC_MOD_FW:

    (...)

    def __init__(
            self,
            env: GymEnv,
            learning_rate: float = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 1000,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            ent_coef: Union[str, float] = "auto",
            use_her: bool = True,
            n_critics: int = 2,
            ignore_dones_for_qvalue: bool = False,
            action_scale_factor: float = 1.0,
            log_obs_step: bool = False,
            log_act_step: bool = False,

            fwd = {}
    ):
```

Now you can instanciate your forward model and the training data buffer. 

Typically, you want the forward model and training data to be attributes of your algorithm instance. In this case, you can simply add the following attributes at the end of your algorithms initialization:

```
class CLEANSAC_MOD_FW:

    (...)

    def __init__(
            self,
            env: GymEnv,
            learning_rate: float = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 1000,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            ent_coef: Union[str, float] = "auto",
            use_her: bool = True,
            n_critics: int = 2,
            ignore_dones_for_qvalue: bool = False,
            action_scale_factor: float = 1.0,
            log_obs_step: bool = False,
            log_act_step: bool = False,

            fwd = {}
    ):

        (...)

        
        """
        Forward model initialization
        """
        self.fwd = fwd
        self.forward_model = ForwardNetEnsemble(self.fwd, self.env, DeterministicForwardNetwork)
        self.fw_optimizer = [torch.optim.Adam(model.parameters(), lr=self.learning_rate) for model in self.forward_model.ensemble]
```

## Find the appropriate place to collect training data

Training data consists of triples `(observation, action, next_observation, reward)`, which you want to push into your training data buffer.

Those triples are typically available right after an `env.step(action)` was performed, and BEFORE the `last_observation` is overwritten by a new observation.

In `cleansac`, an appropriate place is directly in the `step_env()` method, right before the `last_observation` is overwritten. You collect training data by your forward models corresponding method `collect_training_data`:

```
    def step_env(
            self,
            callback: BaseCallback
    ):

        (...)

        # Collect data for the forward model
        self.forward_model.collect_data(self._last_obs, action, new_obs, rewards)

        self._last_obs = new_obs

        (...)
```
## Train your model

A good place to call your forward models `train` method is typically in your algorithms `learn` method (right after calling the RL-algorithms own `train` method).

Everytime the forward models `train` method is called, it is (re-)trained on the whole training data set collected so far. It would be very inefficient to do this after every step, because the gradient computation and backpropagation is rather costly.

Instead, we recommend to only (re-)train your forward model every n'th step:

```
    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval=None,
    ):
        (...)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

                """
                Forward model training
                """
                if self.num_timesteps % self.fwd['train_every_n_data'] == 0:
                    self.forward_model.train(self.fw_optimizer)
               
                    self.logger.record('fwd/train_loss', self.forward_model.get_average_loss(fw_data_loader))

        (...)

```

## Check your implementation
To verify that your implementation is working correctly, you can start a training session with your newly implemented Forward model by running the following command:
`python src/main.py algorithm=cleansac_mod_fw env=FetchReach-v2`.
During training, you should observe that the `fwd/train_loss metric` decreases over time, indicating that the model is learning effectively. 

## Save your forward model

To save your forward model for future use, specify your desired file storage location by filling in the `model_save_path` field within the `cleansac_mod_fw_yaml`. Additionally, ensure you invoke either the `savemodel()` or `savestatedict()` method before completing the training of your reinforcement learning algorithm.


### The difference of `save_model()` and `save_state_dict()`
Within `utils/forward_models.py`, these methods function as follows:
```
    def save_model(self, model_name):
        torch.save(self, os.path.join(self.cfg['model_save_path'], f'{model_name}.pt'))

    def save_state_dict(self, state_dict_name):
        torch.save(self.state_dict(), os.path.join(self.cfg['model_save_path'], f'{state_dict_name}.pt'))
```
The `save_state_dict()` method saves only the `state_dict`, which is a Python dictionary mapping each layer to its parameter tensor. This method is advantageous as it significantly reduces file size and is the **officially recommended approach by PyTorch**.



Conversely, `save_model()` saves the entire model, resulting in a larger file size. A drawback of this method is that the serialized model is tied to the specific classes and directory structure present at the time of saving.

We strongly encourage you to use `save_state_dict()` to save your models. For more information on saving and loading models or state dictionaries, see the [documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html) of PyTorch.  

### Example implementation
```
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1
    ):
        
        (...)
            
        # use eiter this    
        self.forward_model.save_model('your_models_name')
        # or this
        self.forward_model.save_state_dict('your_state_dicts_name')

        callback.on_training_end()

        return self
```

# Functionalities of implemented forward models

This concludes the tutorial, providing you with additional explanations. Scilabrl currently implements two fully connected multi-layer perceptron (MLP) classes:

1. `DeterministicForwardModel` is a point estimator for the next observation.
2. `ProbabilisticForwardMLENetwork` learns the paramenters `mu` and `theta` of a normal distribution over the predicted next observation.

## Training

The models have a `train` method, that expects an optimizer and a dataloader as input:

```
def train(self, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader):
    """
    Trains the model using the provided optimizer and dataloader.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer to be used for training (e.g., Adam, SGD).
    dataloader : torch.utils.data.DataLoader
        A PyTorch dataloader providing batches of input data and target labels.
    epochs : int
        The number of training epochs to run.

    Returns:
    --------
    None
    """
```

## Inference (use model to make predictions)

Inference in forward models involves using the trained models to make predictions based on new data. When these models are designed to predict the change in state, such as `next_observation - observation`, they are particularly well-suited for understanding the transitions or dynamics within a system. To perform inference correctly, you should utilize the models' `predict` method. This method is specifically crafted to compute the accurate predictions by incorporating the learned relationships from training. Avoid relying on the output of a simple forward pass through the network for inference, as it may not fully capture the nuances of the trained prediction differences, and thus might lead to inaccurate results.

```
def predict(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Predicts the next observation based on the given observation and action.

    Parameters:
    -----------
    observation : torch.Tensor
        A tensor representing the current observation (state) of the environment.
    action : torch.Tensor
        A tensor representing the action taken in the current state.

    Returns:
    --------
    torch.Tensor
        A tensor representing the predicted next observation (next state).
    """
```

## Pretraining the Forward Model

Pretraining the forward model allows it to learn an initial approximation of the environment's dynamics before further training.

To pretrain your model, call the `pre_train_model()` function after initializing the forward model. This function optimizes the model's parameters before it is used in a reinforcement learning setting.


```
"""
Forward model initialization
"""
self.fwd = fwd
self.forward_model = ForwardNetEnsemble(self.fwd, self.env, DeterministicForwardNetwork)
self.fw_optimizer = [torch.optim.Adam(model.parameters(), lr=self.learning_rate) for model in self.forward_model.ensemble]

self.forward_model.pre_train_model(self.fw_optimizer)
```