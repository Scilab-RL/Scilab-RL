---
layout: default
title: Add new environments
parent: Wiki
has_children: false
nav_order: 2
---

In this tutorial, you'll learn how to add a new custom _MuJoCo_ environment. It is divided into four steps:

1. Choose and test a base-environment
2. Copy and rename the base-environment
3. Register and test the new environment
4. Modify the new environment

As an example, we'll add a 1-DOF (degree of freedom) version of the `FetchReach-v2` environment by copying the `Blocks` environment.

# Choose and test a base-environment

As a basis, use an environment that is as close as possible to the new environment you want to develop. A good start for a gripper-based environment is the gymnasium-robotics version of `FetchReach`. You could also choose to base your new env on one of our [custom environments](Environments-Overview). 

In our example, we'd like to create a very easy environment that is like the `FetchReach-v2` environment, but the robot only has one degree of freedom (DOF). As a basis, we choose the `BlocksEnv` environment from our custom environments, because it already inherits from the `MujocoFetchEnv`.

Run your favorite algorithm with this base environment and make sure it works as intended. It is also a good idea to take notes about the learning performance (how many training steps are required to achieve success). In our case we run `python src/main.py env=Blocks-o0-gripper_random-v1 algorithm=cleansac wandb=0`, which uses the default algorithm SAC with HER, and achieves 1.0 `eval/success_rate` after the third epoch.

# Copy and rename the base-environment

Create a copy of your base-environment and add it to the `src/custom_envs` folder in the repository. To do that, create a new subfolder for your new environment under `src/custom_envs`.
(The `Reach1DOF` environment has meanwhile been added to the repository, because it is good for testing simple algorithms.
You'll have to delete it to follow this tutorial.)
For our example, we copy the `src/custom_envs/blocks` folder and paste it to the `src/custom_envs` directory with the name `reach1dof`. We also rename the copied

- `blocks_env.py` to `reach1dof_env.py`
- in `reach1dof_env.py`: `BlocksEnv` to `Reach1DOFEnv`


## Side note: entry points
The original `MujocoFetchReachEnv` environment uses _entry points_.
Using an entry point class is the preferred way of implementing different variations for each environment. For example, the `MujocoFetchEnv` has different entry points for `MujocoFetchReachEnv` and `MujocoFetchPushEnv`.
You find these in `reach.py` and `push.py` respectively in `/home/username/miniforge3/envs/scilabrl/lib/python3.11/site-packages/gymnasium_robotics/envs/fetch/`.

# Register and test the new environment

Then you need to register your new environment. You can do this by adding it to the `src/custom_envs/register_envs.py` file. For our example environment, we add 
```
register(id='Reach1DOF-v0',
         entry_point='custom_envs.reach1dof.reach1dof_env:Reach1DOFEnv',
         max_episode_steps=50)
```

Note that the `-vN` suffix is mandatory for each environment name, where `N` is the version number. We can now call the copied environment with `python src/main.py env=Reach1DOF-v0 wandb=0` and receive a `TypeError: Reach1DOFEnv.__init__() missing 2 required positional arguments: 'n_objects' and 'gripper_goal'`. That happened because the original `Blocks` env requires keyword arguments that are specified with the environment ID. E.g. to launch the `Blocks` env with `n_objects=0` and `gripper_goal='gripper_random'`, specify `env=Blocks-o0-gripper_random-v1`. As a quick fix, we set default arguments for the two _kwargs_ in `reach1dof_env.py`:
```
def __init__(self, n_objects=0, gripper_goal='gripper_random', distance_threshold=0.05, reward_type='sparse', model_xml_path=MODEL_XML_PATH):
```

Now try whether your copy of the base environment is running as intended, in the same way as the original one. For our example, we run `python src/main.py env=Reach1DOF-v0 algorithm=cleansac wandb=0` again. It should achieve 100% `eval/success_rate` after 3 to 4 epochs.

# Modify the new environment
We'd like to modify the environment so that the robot only has to move forwards or backwards.
Therefore, we have to change some methods:

## __init__(...)
The goal should only contain __one value__ and always be on an axis that the robot can reach by only moving forwards or backwards. So at first we change:
```
self.goal_size = self.n_objects * 3
if self.gripper_goal != 'gripper_none':
    self.goal_size += 3
```
to
```
self.goal_size = 1
```

## _sample_goal()
Sample goal had to handle all the different blocks, but now we only want one value for the x-coordinate of the gripper. So we delete the whole `_sample_goal()` method and replace it with:
```
def _sample_goal(self):
    goal = np.array([self.initial_gripper_xpos[0]])
    goal[0] += self.np_random.uniform(-self.target_range, self.target_range)
    return goal.copy()
```
So now the goal is an x-coordinate between the grippers original x-position + `self.target_range` and the original x-position - `self.target_range`.

We can now run `python src/main.py wandb=0 env=Reach1DOF-v0 render=display` to see the modified version, but we will not see the goal. For that, we first have to change the `_render_callback()` which visualizes the goal.

> 💡 You can simply end a visualized MuJoCo experiment by pressing _ESC_.

## _render_callback()
Change it to
```
def _render_callback(self):
    goal_pos = self.initial_gripper_xpos.copy()
    goal_pos[0] = self.goal[0]
    site_id = self._model_names.site_name2id['gripper_goal']
    self.model.site_pos[site_id] = goal_pos
```
We need three coordinates for the goal-visualization-site (a site in MuJoCo is an object that does not participate in collisions). That is why we copy the initial gripper position and only change the x-coordinate. Then we put the goal-visualization site to that position. You should now be able to see the goal change only its x-position with `python src/main.py wandb=0 env=Reach1DOF-v0 render=display`.

Now we also need to make the robot move only on the x-axis:

## _set_action(action)
`_set_action()` is not defined in the `BlocksEnv`/`Reach1DOFEnv`. We need to overwrite the `_set_action()` method that we inherit from the `MujocoFetchEnv`. Therefore, we add this method to the `Reach1DOFEnv`:
```
def _set_action(self, action):
    action = np.concatenate([action, np.zeros(3)])
    super()._set_action(action)
```

The `MujocoFetchEnv._set_action(action)` expects an action with `shape==(4,)`, where the first three values change the x-, y- and z-position of the gripper and the fourth value opens or closes the gripper. We just set everything except the x-value to zero and then call the method.

Running `python src/main.py wandb=0 env=Reach1DOF-v0 render=display` again gives us an `AssertionError`. That is because we have not adjusted `self.action_space` yet. The action space tells the algorithm how many values each action has and in what range these values are expected. To set the action space to only one value, add the line 
```
self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype="float32")
```
at the end of `__init__()`, after the call to `super().__init__()`. Also add
```
from gymnasium import spaces
```
to the imports.

`python src/main.py wandb=0 env=Reach1DOF-v0 algorithm=cleansac render=display` should now show the robot moving with only one DOF. SAC + HER should be able to solve this environment practically immediately.

> ⚠️ Of course these changes only lead to a first version of the new environment. You'll also need to change the _docstrings_ and remove unused code (e.g. the code that sets the position of the blocks, as we do not use blocks).

> 💡 You may also want to copy and modify the environment XML for MuJoCo environments. You can find the XMLs in the `custom_envs/assets` folder. They are specified with the `model_xml_path`.

Finally, you could find out how fast your environment can be solved with [hyperparameter optimization](Hyperparameter-optimization) or check whether it runs with all algorithms by adding it to the `ENVS` in `test_algos()` in `scripts/run_smoke_tests.sh` ([more on smoke tests](Smoke-tests)).

Further details can be found in [add environment to MakeDictObs wrapper](Add-environment-to-MakeDictObs-wrapper)