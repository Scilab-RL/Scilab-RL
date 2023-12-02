import torch


class RANDOM:
    """
    This algorithm samples randomly from the action space.
    """

    def __init__(self, env):
        self.env = env
        self.num_timesteps = 0
        self.logger = None
        self.env.reset()

    def learn(self, total_timesteps, callback, log_interval):
        """
        The learning loop.
        The agent carries out a random action and receives an observation and a reward.
        """
        callback.init_callback(model=self)

        while self.num_timesteps < total_timesteps:
            action = self.env.action_space.sample()
            obs, rewards, done, info = self.env.step([action])
            if done:
                callback.on_rollout_start()
                self.env.reset()
            if not callback.on_step():
                return
            self.num_timesteps += 1

        callback.on_training_end()

    @classmethod
    def load(cls, path, env):
        model = cls(env=env)
        return model

    def save(self, path):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps"]:
            del data[to_exclude]
        torch.save(data, path)

    def predict(self, obs, state, deterministic, episode_start):
        return [self.env.action_space.sample()], state

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env
