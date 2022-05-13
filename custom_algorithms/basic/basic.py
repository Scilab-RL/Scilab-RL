import torch as th
import numpy as np


def prepare_obs(obs):
    return th.cat([th.tensor(o.copy().flatten(), dtype=th.float32) for o in obs.values()])


mae = th.nn.L1Loss(reduction='sum')


class BASIC:
    """
    This is the most basic algorithm that works with our framework.
    """
    # possible todos:
    # TODO use separate env for evaluation?
    # TODO policy saving and loading
    # TODO nn backprop
    # TODO do not use the callback AT ALL (modify train.py)
    def __init__(self, env, net_arch=None, noise_factor=0.1, learning_rate=0.001):
        self.env = env
        self.logger = None
        self.num_timesteps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_obs = self.env.reset()

        # For Deep-q-learning:
        self.lr = learning_rate
        self.noise_factor = noise_factor
        self.gamma = 0.95

        # Networks
        self.actor = self._setup_nn(net_arch)
        self.a_opt = th.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = self._setup_critic(net_arch)
        self.c_opt = th.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.target = self._setup_critic(net_arch)
        self.target.load_state_dict(self.critic.state_dict())

        # To get from callback
        self.eval_freq = 0
        self.n_eval_episodes = 0
        self.render = False
        self.early_stop_last_n = 1
        self.early_stop_threshold = 1
        self.history = []

    def set_logger(self, logger):
        self.logger = logger

    def get_settings_from_callback(self, callback):
        self.eval_freq = callback.eval_freq
        self.n_eval_episodes = callback.n_eval_episodes
        self.render = callback.render_train == 'display' or callback.render_test == 'display'  # TODO more sophisticated rendering
        assert callback.early_stop_data_column == 'test/success_rate', "early_stop_data_column has to be test/" \
                                                                       "success_rate. It is the only one available."
        self.early_stop_last_n = callback.early_stop_last_n
        self.early_stop_threshold = callback.early_stop_threshold

    def learn(self, total_timesteps, callback, log_interval):
        self.get_settings_from_callback(callback[0])
        while self.num_timesteps < total_timesteps:
            action = self._get_action(self._last_obs, deterministic=False)
            obs, rewards, done, info = self.env.step(action)
            if self.render:
                self.env.render()
            self._train(self._last_obs, obs, rewards)
            self._last_obs = obs
            self.num_timesteps += 1
            if done:
                self._last_obs = self.env.reset()
            if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
                if self.evaluate():
                    return

    def _train(self, last_obs, obs, reward):
        reward = abs(reward) * 10  # Net minimizes reward, reward closer to 0 is better
        self.logger.record_mean('reward', reward)
        # critic loss
        q_last = self.critic(th.cat([self.actor(prepare_obs(last_obs)), prepare_obs(last_obs)]))
        q_now = self.target(th.cat([self.actor(prepare_obs(obs)), prepare_obs(obs)]))
        critic_loss = mae(q_last, (reward + self.gamma * q_now))
        self.logger.record_mean('critic_loss', critic_loss.item())
        # optimize critic
        self.c_opt.zero_grad()
        critic_loss.backward()
        self.c_opt.step()
        # actor loss
        actor_loss = self.critic(th.cat([self.actor(prepare_obs(last_obs)), prepare_obs(last_obs)]))  # q_last
        self.logger.record_mean('actor_loss', actor_loss.item()) # same as q
        # optimize actor
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()
        if self.num_timesteps % 10 == 0:
            self.target.load_state_dict(self.critic.state_dict())

    def evaluate(self):
        successful_episodes = 0
        for _ in range(self.n_eval_episodes):
            done = False
            while not done:
                action = self._get_action(self._last_obs, deterministic=True)
                obs, reward, done, _info = self.env.step(action)
                if self.render:
                    self.env.render()
                self._last_obs = obs
                success = _info['is_success']
                if success:
                    successful_episodes += 1
                    done = True
                    self.env.reset()
        succ_rate = successful_episodes / self.n_eval_episodes
        self.logger.record('test/success_rate', succ_rate)
        self.logger.dump(step=self.num_timesteps)
        # early stopping
        self.history.append(succ_rate)
        if len(self.history) >= self.early_stop_last_n:
            avg = sum(self.history[:-self.early_stop_last_n])/self.early_stop_last_n
            if avg >= self.early_stop_threshold:
                self.logger.info(f"Early stop threshold for test/success_rate met: "
                                 f"Average over last {self.early_stop_last_n} evaluations is {avg} "
                                 f"and threshold is {self.early_stop_threshold}. Stopping training.")
                return True
        return False

    def load(self, path):
        pass

    def save(self, path):
        pass

    def _get_action(self, obs, deterministic):
        obs = prepare_obs(obs)
        with th.no_grad():
            action = self.actor(obs).detach().numpy()
        if not deterministic:
            action += self.noise_factor * (np.random.normal(size=len(action))-0.5)
        action = np.clip(action, -1, 1)
        return action

    def _setup_nn(self, net_arch):
        input_dim = np.sum([space.shape for space in self.observation_space.spaces.values()])
        output_dim = self.action_space.shape[0]
        if net_arch is None:
            return th.nn.Sequential(th.nn.Linear(input_dim, output_dim))
        modules = [th.nn.Linear(input_dim, net_arch[0]), th.nn.ReLU()]
        for i in range(len(net_arch) - 1):
            modules.append(th.nn.Linear(net_arch[i], net_arch[i + 1]))
            modules.append(th.nn.ReLU())
        last_layer_dim = net_arch[-1]
        modules.append(th.nn.Linear(last_layer_dim, output_dim))
        nn = th.nn.Sequential(*modules)
        return nn

    def _setup_critic(self, net_arch):
        input_dim = np.sum([space.shape for space in self.observation_space.spaces.values()]) + self.action_space.shape[0]
        output_dim = 1
        if net_arch is None:
            return th.nn.Sequential(th.nn.Linear(input_dim, output_dim))
        modules = [th.nn.Linear(input_dim, net_arch[0]), th.nn.ReLU()]
        for i in range(len(net_arch) - 1):
            modules.append(th.nn.Linear(net_arch[i], net_arch[i + 1]))
            modules.append(th.nn.ReLU())
        last_layer_dim = net_arch[-1]
        modules.append(th.nn.Linear(last_layer_dim, output_dim))
        nn = th.nn.Sequential(*modules)
        return nn
