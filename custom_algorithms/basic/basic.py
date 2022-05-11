import torch as th
import numpy as np


def prepare_obs(obs):
    return th.cat([th.tensor(o.copy().flatten(), dtype=th.float32) for o in obs.values()])


class BASIC:
    """
    This is the most basic algorithm that works with our framework.
    """
    # possible todos:
    # TODO use separate env for evaluation
    # TODO policy saving and loading
    # TODO nn backprop
    # TODO do not use the callback AT ALL (modify train.py)
    def __init__(self, env, net_arch=None, noise_factor=0.3, learning_rate=0.0003):
        self.env = env
        self.logger = None
        self.num_timesteps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_obs = self.env.reset()
        self.gamma = 0.95
        self.lr = learning_rate
        self.loss_fn = th.nn.MSELoss(reduction='sum')
        self.actor = self._setup_nn(net_arch)
        self.critic = self._setup_critic(net_arch)
        self.target = self._setup_critic(net_arch)
        self.target.load_state_dict(self.critic.state_dict())
        self.noise_factor = noise_factor
        self.eval_freq = 0
        self.n_eval_episodes = 0
        self.render = False

    def set_logger(self, logger):
        self.logger = logger

    def learn(self, total_timesteps, callback, log_interval):
        callback = callback[0]
        self.eval_freq = callback.eval_freq
        self.n_eval_episodes = callback.n_eval_episodes
        while self.num_timesteps < total_timesteps:
            action = self._get_action(self._last_obs, deterministic=False)
            obs, rewards, done, info = self.env.step(action)
            if self.render:
                self.env.render()
            self._train(self._last_obs, obs, rewards)
            self._last_obs = obs
            self.num_timesteps += 1
            if done:
                self.env.reset()
            if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
                self.evaluate()

    def _train(self, last_obs, obs, reward):
        self.logger.record_mean('reward', reward)
        q_last = self.critic(th.cat([self.actor(prepare_obs(last_obs)), prepare_obs(last_obs)]))
        q_now = self.target(th.cat([self.actor(prepare_obs(obs)), prepare_obs(obs)]))
        self.logger.record_mean('q', q_now.item())
        critic_loss = self.loss_fn(q_last, (reward + self.gamma * q_now))
        actor_loss = q_last
        self.logger.record_mean('critic_loss', critic_loss.item())
        self.logger.record_mean('actor_loss', actor_loss.item())
        self.critic.zero_grad()
        self.actor.zero_grad()
        if np.random.rand() > 0.5:
            actor_loss.backward()
            with th.no_grad():
                for param in self.actor.parameters():
                    param -= self.lr * param.grad
        else:
            critic_loss.backward()
            with th.no_grad():
                for param in self.critic.parameters():
                    param -= self.lr * param.grad
        if self.num_timesteps % 50 == 0:
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
        self.logger.record('test/success_rate', successful_episodes / self.n_eval_episodes)
        self.logger.dump(step=self.num_timesteps)
        #print([p for p in self.actor.parameters()][0])
        # TODO early stopping

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
