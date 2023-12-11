---
layout: default
title: Add environment to MakeDictObs wrapper
parent: Wiki
has_children: false
nav_order: 2
---

Hindsight Experience Replay (HER) can radically improve the sample efficiency,
but it is only applicable to an environment if it fulfills the following requirements:
1. It is goal-conditioned 
2. It has a `DictObservationSpace` with "observation", "achieved_goal" and "desired_goal".
3. It has a `compute_reward(achieved_goal, desired_goal, infos) -> reward` function, that calculates the correct
reward given the achieved goal and desired goal.

Not all environments, e.g. the MetaWorld environments, have the option to provide dict observations 
or compute the reward retrospectively.
Therefore, we implement this functionality with the `MakeDictObs` Wrapper, located in `utils/custom_wrappers.py`.
You can take a look at the implementations that are already there, e.g. for `MetaW-reach-v2-sparse` 
which originally had the name `reach-v2-goal-observable`.

To add an environment to the `MakeDictObs` wrapper, you'll have to follow these steps:

# Find out which part of the observation is relevant for the goal computation.

Look at the environment implementation and find out how the reward is computed.
This is often implemented in a function called `compute_reward`.

For most environments, the condition for success is that _something_ has to be at certain _position_.
This means that the `desired_goal` is the position where the object should be and the `achieved_goal` is 
the actual position of the object. If the reward is dense, it is often the distance between those two,
and if it is sparse, it is whether the distance is below a certain threshold.

It is sufficient to implement only a version with sparse rewards, because they are better suited for SAC with HER.


###  For Metaworld environments the observation always has the same structure:

They all have shape (39,) and are composed like this:

`obs = np.hstack((curr_obs, self._prev_obs, pos_goal))`

That means
- obs[0:18] = observation
- obs[18:36] = observation from last step
- obs[36:39] = goal

observation is composed like this: `np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))`
with 
- pos_hand: xyz of the endeffector
- gripper_distance_apart: one value that reflects how open the gripper is 
- obs_obj_padded: 14 zeros that are sometimes filled with object info. If it is fully filled, it will be
                     `[obj_1_xyz (3 values), obj_1_quat (4 values), obj_2_xyz (3 values), obj_2_quat (4 values)]`

# Add an if-statement to check for environment name(s)
The dict-obs conversion you will implement will probably only be valid for your environment and maybe a 
few similar environments.
If your environment is similar to an environment for which a conversion is already available,
you might be able to reuse that implementation by just adding your environment to its if-statement.
Otherwise, add an if-statement to check for your environment name in `MakeDictObs.__init__()`.
Your code will be inside this if-statement.

# Define the observation-space
You'll have to overwrite the observation-space to be a Dictionary-observation-space. Chances are that you can
use the original observation-space for that. As an example, here is how it could be done for `reach-v2`:

```
low = self.env.observation_space.low
high = self.env.observation_space.high
env.observation_space = spaces.Dict(
    dict(
        desired_goal=spaces.Box(
            low=low[-3:], high=high[-3:], dtype="float64"
        ),
        achieved_goal=spaces.Box(
            low=low[:3], high=high[:3], dtype="float64"
        ),
        observation=spaces.Box(
            low=low[3:-3], high=high[3:-3], dtype="float64"
        ),
    )
)
```

In this example, the desired goal consists of the last 3 values of the observation and the achieved goal consists
of the first 3 values of the observation. The rest will be provided as the "observation".

# Define `convert_obs(obs)`
Then you'll need to define a `convert_obs(obs)` function that turns the observation into a dict-observation.
Example for `reach-v2`:
```
def convert_obs(obs):
    ag = obs[:3]
    dg = obs[-3:]
    ob = obs[3:-3]
    return {"observation": ob, "achieved_goal": ag, "desired_goal": dg}
self.obs_to_dict_obs = convert_obs
```
`self.obs_to_dict_obs(obs)` will be called at every step to convert the observations.

# Define `compute_reward` function
For sparse rewards, it is often enough to see if the distance between achieved_goal and desired_goal is
below a threshold. The threshold has to be copied from the original environment reward computation.
```
def compute_reward(achieved_goal, desired_goal, infos):
    distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
    return (distances < 0.05) - 1
```


Don't forget to make the function available to the algorithm with

`self.compute_reward = compute_reward`

Because this is a wrapper, the environment will still use its original `compute_reward` during the rollout, but
if `compute_reward` is called from outside the environment class, the `compute_reward` that you just implemented 
will be called.

# Register your environment
In order to use your environment, you have to register it in gymnasium. To use it with the wrapper, you'll
have to register it with a closure that uses the wrapper when `gym.make(env)` is called. You can also choose
a name for your environment there.
An example on how to do this for the Metaworld environments can be seen at the end of `register_envs.py`.
