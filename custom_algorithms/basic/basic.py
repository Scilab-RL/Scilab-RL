import pickle
import torch as th
import numpy as np


# mean absolute error
mae = th.nn.L1Loss(reduction='sum')


def prepare_obs(obs):
    """
    Creates a flat tensor from the observation dictionary
    """
    return th.cat([th.tensor(o.copy().flatten(), dtype=th.float32) for o in obs.values()])


def create_nn(net_arch, input_dim, output_dim):
    """
    Creates a sequential neural network
    :param net_arch: list, how many nodes in each hidden layer
                     example: net_arch = [16, 16]
                     --> two hidden layers with 16 nodes each
                     If None, no hidden layers
    :param input_dim: int, how many nodes in the input layer
    :param output_dim: int, how many nodes in the output layer

    The activation function after each layer except the last is ReLU.
    """
    if net_arch is None:
        return th.nn.Sequential(th.nn.Linear(input_dim, output_dim))
    modules = [th.nn.Linear(input_dim, net_arch[0]), th.nn.ReLU()]
    for i in range(len(net_arch) - 1):
        modules.append(th.nn.Linear(net_arch[i], net_arch[i + 1]))
        modules.append(th.nn.ReLU())
    last_layer_dim = net_arch[-1]
    modules.append(th.nn.Linear(last_layer_dim, output_dim))
    return th.nn.Sequential(*modules)


class BASIC:
    """
    This is a very basic algorithm that works with our framework.
    It is an on-policy deep-Q-learning agent.
    This means we have
    - an actor-network, which chooses an action given the observation
    - a critic-network, which provides a score for the action given the action and observation
    - a target-network, which is a copy of the critic network for learning stability
    """
    # possible todos:
    # TODO policy saving and loading
    # TODO do not use the callback AT ALL (modify train.py)
    def __init__(self, env, net_arch=None, noise_factor=0.1, learning_rate=0.001):
        self.env = env
        self.logger = None
        self.n_steps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_obs = self.env.reset()

        # For deep-Q-learning:
        self.lr = learning_rate
        self.noise_factor = noise_factor
        self.gamma = 0.95

        # Networks
        n_obs = np.sum([space.shape for space in self.observation_space.spaces.values()])
        n_actions = self.action_space.shape[0]
        self.actor = create_nn(net_arch, n_obs, n_actions)
        self.a_opt = th.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = create_nn(net_arch, n_obs + n_actions, 1)
        self.c_opt = th.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.target = create_nn(net_arch, n_obs + n_actions, 1)
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
        """
        Get settings for the experiment from the callback.
        Currently, train.py does not provide these values directly, so we get them this way.
        """
        self.eval_freq = callback.eval_freq
        self.n_eval_episodes = callback.n_eval_episodes
        self.render = callback.render_train == 'display' or callback.render_test == 'display'  # TODO more sophisticated rendering
        assert callback.early_stop_data_column == 'test/success_rate', "early_stop_data_column has to be test/" \
                                                                       "success_rate. It is the only one available."
        self.early_stop_last_n = callback.early_stop_last_n
        self.early_stop_threshold = callback.early_stop_threshold

    def learn(self, total_timesteps, callback, log_interval):
        """
        The learning loop.
        The agent chooses an action for the environment and receives an observation and a reward.
        After that, it calls _train() to update the actor and critic.
        Every self.eval_freq steps, the policy is evaluated.
        """
        self.get_settings_from_callback(callback[0])
        while self.n_steps < total_timesteps:
            action = self._get_action(self._last_obs, deterministic=False)
            obs, rewards, done, info = self.env.step(action)
            if self.render:
                self.env.render()
            self._train(self._last_obs, obs, rewards)
            self._last_obs = obs
            self.n_steps += 1
            if done:
                self._last_obs = self.env.reset()
            if self.eval_freq > 0 and self.n_steps % self.eval_freq == 0:
                if self.evaluate():
                    return

    def _train(self, last_obs, obs, reward):
        """
        Train the network with the new reward and observation from the environment.
        On-policy deep-Q-learning.
        """
        reward = abs(reward) * 10  # reward closer to 0 is better
        self.logger.record_mean('reward', -reward)
        # critic loss
        q_last = self.critic(th.cat([self.actor(prepare_obs(last_obs)), prepare_obs(last_obs)]))
        q_now = self.target(th.cat([self.actor(prepare_obs(obs)), prepare_obs(obs)]))
        critic_loss = mae(q_last, (reward + self.gamma * q_now))  # temporal difference error
        self.logger.record_mean('critic_loss', critic_loss.item())
        # optimize critic
        self.c_opt.zero_grad()
        critic_loss.backward()
        self.c_opt.step()
        # actor loss
        actor_loss = self.critic(th.cat([self.actor(prepare_obs(last_obs)), prepare_obs(last_obs)]))  # q_last
        self.logger.record_mean('actor_loss', actor_loss.item())
        # optimize actor
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()
        if self.n_steps % 10 == 0:  # update the weights of the target network every 10 steps
            self.target.load_state_dict(self.critic.state_dict())

    def evaluate(self):
        """
        Evaluates the policy for self.n_eval_episodes episodes.
        Checks whether the threshold for early stopping is met.
        """
        successful_episodes = 0
        for _ in range(self.n_eval_episodes):
            self._last_obs = self.env.reset()
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
        suc_rate = successful_episodes / self.n_eval_episodes
        self.logger.record('test/success_rate', suc_rate)
        self.logger.dump(step=self.n_steps)
        # early stopping
        self.history.append(suc_rate)
        if len(self.history) >= self.early_stop_last_n:
            avg = sum(self.history[-self.early_stop_last_n:])/self.early_stop_last_n
            if avg >= self.early_stop_threshold:
                self.logger.info(f"Early stop threshold for test/success_rate met: "
                                 f"Average over last {self.early_stop_last_n} evaluations is {avg} "
                                 f"and threshold is {self.early_stop_threshold}. Stopping training.")
                p = self.logger.get_dir() + "/best.zip"
                self.logger.info(f"Saving policy to {p}")
                self.save(path=p)
                return True
        return False

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = pickle.load(open(path, "rb"))
        for k in loaded_dict:
            if k not in ["actor_state", "critic_state"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.actor.load_state_dict(loaded_dict["actor_state"])
        model.critic.load_state_dict(loaded_dict["critic_state"])
        model.target.load_state_dict(loaded_dict["critic_state"])
        return model

    def save(self, path):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "n_steps", "history", "_last_obs", "render",
                           "actor", "critic", "target"]:
            del data[to_exclude]
        # save network parameters
        data["actor_state"] = self.actor.state_dict()
        data["critic_state"] = self.critic.state_dict()
        # no need to save the target-network state, because it is a copy of the critic network
        pickle.dump(data, open(path, "wb"))

    def _get_action(self, obs, deterministic):
        """
        Get action from the actor network.
        If the action should not be deterministic, add noise with intensity self.noise_factor.
        """
        obs = prepare_obs(obs)
        with th.no_grad():
            action = self.actor(obs).detach().numpy()
        if not deterministic:
            action += self.noise_factor * (np.random.normal(size=len(action))-0.5)
        action = np.clip(action, -1, 1)
        return action
