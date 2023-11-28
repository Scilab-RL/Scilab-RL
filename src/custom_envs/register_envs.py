"""
All custom environments must be registered here, otherwise they won't be found.
"""
from gymnasium.envs.registration import register
import highway_env
RESET = R = "r"  # Initial Reset position of the agent
GOAL = G = "g"
COMBINED = C = "c"  # These cells can be selected as goal or reset locations


OPEN = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]
OPEN_DIVERSE_G = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, G, 1],
    [1, G, G, G, G, G, 1],
    [1, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

OPEN_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, C, C, C, C, C, 1],
    [1, C, C, C, C, C, 1],
    [1, C, C, C, C, C, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

def _merge(a, b):
    a.update(b)
    return a

def register_custom_envs():
    for n_objects in range(5):
        for gripper_goal in ['gripper_none', 'gripper_random', 'gripper_above']:
            if gripper_goal != 'gripper_random' and n_objects == 0:  # Disallow because there would be no goal
                continue
            distance_threshold = 0.05  # was originally 0.05
            register(id=f'Blocks-o{n_objects}-{gripper_goal}-v1',
                     entry_point='custom_envs.blocks.blocks_env:BlocksEnv',
                     kwargs={'n_objects': n_objects, 'gripper_goal': gripper_goal, 'distance_threshold': distance_threshold},
                     max_episode_steps=max(50, 50*n_objects))

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        register(id=f'AntGymnasiumMod{suffix}-dt15-openDGR-v0',
            entry_point='custom_envs.ant.ant_env:AntGymMod',
            kwargs = _merge(
                {
                    "distance_threshold": 1.5,
                    "maze_map": OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps = 700,
            )
        register(id=f'AntGymnasiumMod{suffix}-dt15-openDG-v0',
             entry_point='custom_envs.ant.ant_env:AntGymMod',
             kwargs=_merge(
                 {
                     "distance_threshold": 1.5,
                     "maze_map": OPEN_DIVERSE_G,
                 },
                 kwargs,
             ),
             max_episode_steps=700,
             )


    register(id='Reach1DOF-v0',
             entry_point='custom_envs.reach1dof.reach1dof_env:Reach1DOFEnv',
             max_episode_steps=50)

    for n_objects in range(3):
        register(id=f'Hook-o{n_objects}-v1',
                 entry_point='custom_envs.hook.hook_env:HookEnv',
                 kwargs={'n_objects': n_objects},
                 max_episode_steps=max(50, 100 * n_objects))

        register(id=f'ButtonUnlock-o{n_objects}-v1',
                 entry_point='custom_envs.button_unlock.button_unlock_env:ButtonUnlockEnv',
                 kwargs={'n_buttons': n_objects+1},
                 max_episode_steps=max(50, 50*n_objects))

    register(
        id='parking-limited-v0',
        entry_point='highway_env.envs:ParkingEnv',
        max_episode_steps=100,
    )
