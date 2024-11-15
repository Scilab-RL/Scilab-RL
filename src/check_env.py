import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as check_env_sb3
from src.custom_envs.register_envs import register_custom_envs
from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv


# @hydra.main(config_name="main", config_path="../../../conf", version_base="1.1.2")
def main():
    environment = MoonlanderWorldEnv()
    # check_env(environment)

    observation, info = environment.reset()
    environment.render()

    terminated = False
    while not terminated:
        action = (
            environment.action_space.sample()
        )  # agent policy that uses the observation and info*.py
        observation, reward, terminated, truncated, info = environment.step(action)
        environment.render()

    environment.close()


if __name__ == "__main__":
    register_custom_envs()
    print("registered custom envs")

    env = gym.make("MetaEnv-pretrained-human-v0")
    check_env(env)
    check_env_sb3(env)
    main()
