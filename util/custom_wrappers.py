import gym


class DisplayWrapper(gym.Wrapper):
    """
    Display the environment every nth epoch. You must specify one (and only one) of epoch_steps or epoch_episodes.
    :param env: the environment
    :param every_nth_epoch: the environment will be displayed every nth epoch
    :param epoch_steps: how many steps each epoch consists of
    :param epoch_episodes: how many episodes each epoch consists of
    """
    def __init__(self, env, every_nth_epoch, epoch_steps=None, epoch_episodes=None):
        super().__init__(env)
        self.every_nth_epoch = every_nth_epoch
        n_specified = sum([x is not None for x in [epoch_steps, epoch_episodes]])
        assert n_specified == 1, "Must specify either epoch_steps or epoch_episodes but not both."
        self.epoch_steps = epoch_steps
        self.epoch_episodes = epoch_episodes
        self.step_id = 0
        self.episode_id = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.epoch_steps is not None and self.step_id % (self.every_nth_epoch * self.epoch_steps) < self.epoch_steps:
            self.env.render(mode='human')
        if self.epoch_episodes is not None and self.episode_id % (self.every_nth_epoch * self.epoch_episodes) < self.epoch_episodes:
            self.env.render(mode='human')
        # increment steps and episodes
        self.step_id += 1
        if done:
            self.episode_id += 1
        return obs, reward, done, info
