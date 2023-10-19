# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import gymnasium as gym
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
        self.fc1 = nn.Linear(obs_shape + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        # log_std = torch.tanh(log_std)
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob


def flatten_obs(obs):
    observation, ag, dg = obs['observation'], obs['achieved_goal'], obs['desired_goal']
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).to(device)
    if isinstance(ag, np.ndarray):
        ag = torch.from_numpy(ag).to(device)
    if isinstance(dg, np.ndarray):
        dg = torch.from_numpy(dg).to(device)
    return torch.cat([observation, ag, dg], dim=1).to(dtype=torch.float32).detach().clone()


class MYSACHER:
    """
    Custom version of SAC adapted from CleanRL
    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
    https://docs.cleanrl.dev/rl-algorithms/sac

    Modified so that it works with dict observations and can use the HER ReplayBuffer from SB3.

    :param env: a gymnasium environment with discrete action space
    :param q_lr: the learning rate for the critic network
    :param policy_lr: the learning rate for the policy network
    :param autotune: whether to automatically tune the entropy coefficient alpha
    :param alpha: the entropy regularization coefficient alpha
    :param learning_starts: start the training after learning_starts steps
    :param gamma: the discount factor gamma
    :param policy_frequency: train the policy every policy_frequency steps
    :param target_network_frequency: update the target network every target_network_frequency steps
    :param tau: target network smoothing coefficient
    :param batch_size: size of the batches sampled from the replay buffer
    :param buffer_size: size of the replay buffer memory
    """
    def __init__(self,
                 env: GymEnv,
                 q_lr: float,
                 policy_lr: float,
                 autotune: bool,
                 alpha: float,
                 buffer_size: int,
                 learning_starts: int,
                 batch_size: int,
                 gamma: float,
                 policy_frequency: int,
                 target_network_frequency: int,
                 tau: float):
        self.env = env
        assert isinstance(self.env.action_space, gym.spaces.Box), "only continuous action space is supported"

        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        self.actor = Actor(self.env).to(device)
        self.qf1 = SoftQNetwork(self.env).to(device)
        self.qf2 = SoftQNetwork(self.env).to(device)
        self.qf1_target = SoftQNetwork(self.env).to(device)
        self.qf2_target = SoftQNetwork(self.env).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)

        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

        self.env.observation_space.dtype = np.float32

        # self.rb = HerReplayBuffer(
        #     buffer_size=buffer_size,
        #     observation_space=self.env.observation_space,
        #     action_space=self.env.action_space,
        #     device=device,
        #     handle_timeout_termination=False,
        #     env=self.env,
        # )

        self.rb = DictReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=device,
        )

        self.num_timesteps = 0
        self.n_updates = 0
        self.logger = None
        self.callback = None

    def learn(self, total_timesteps: int, callback, log_interval):
        """
        learn to get a good reward for the environment
        :param total_timesteps: the maximum number of timesteps to train the agent
        :param callback: a Callback or CallbackList to call every step, e.g. EvalCallback
        """
        self.callback = callback
        self.callback.init_callback(self)

        obs = self.env.reset()
        for global_step in range(total_timesteps):
            self.num_timesteps += 1
            actions, _ = self.predict(obs, state=None, episode_start=None,
                                      deterministic=global_step >= self.learning_starts)

            next_obs, rewards, dones, infos = self.env.step(actions)

            # save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            obs = next_obs.copy()

            if global_step > self.learning_starts:
                self.train(global_step)

            if callback.on_step() is False:
                return

    def train(self, global_step):
        self.n_updates += 1
        data = self.rb.sample(self.batch_size)
        observations = flatten_obs(data.observations)
        next_observations = flatten_obs(data.next_observations)

        if self.autotune:
            with torch.no_grad():
                _, log_pi = self.actor.get_action(observations)
            # alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
            alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            self.logger.record_mean("train/ent_coef_loss", alpha_loss.item(), global_step)
        self.logger.record_mean("train/ent_coef", self.alpha, global_step)

        with torch.no_grad():
            next_state_actions, next_state_log_pi = self.actor.get_action(next_observations)
            qf1_next_target = self.qf1_target(next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.flatten()

        qf1_a_values = self.qf1(observations, data.actions).view(-1)
        qf2_a_values = self.qf2(observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = 0.5 * (qf1_loss + qf2_loss)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        self.logger.record_mean("train/qf1_values", qf1_a_values.mean().item(), global_step)
        self.logger.record_mean("train/qf2_values", qf2_a_values.mean().item(), global_step)
        self.logger.record_mean("train/qf1_loss", qf1_loss.item(), global_step)
        self.logger.record_mean("train/qf2_loss", qf2_loss.item(), global_step)
        self.logger.record_mean("train/critic_loss", qf_loss.item() / 2.0, global_step)
        self.logger.record("train/n_updates", self.n_updates)

        if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                self.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi = self.actor.get_action(observations)
                qf1_pi = self.qf1(observations, pi)
                qf2_pi = self.qf2(observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.logger.record_mean("train/actor_loss", actor_loss.item(), global_step)

        # update the target networks
        if global_step % self.target_network_frequency == 0:
            with torch.no_grad():
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def predict(self, obs, state, episode_start, deterministic):
        if deterministic:
            obs = flatten_obs(obs)
            actions, _ = self.actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        else:
            actions = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])

        return actions, state

    def save(self, path):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "n_updates", "rb",
                           "actor", "qf1", "qf2", "qf1_target", "qf2_target", "callback"]:
            del data[to_exclude]
        # save network parameters
        data["_actor"] = self.actor.state_dict()
        data["_qf1"] = self.qf1.state_dict()
        data["_qf2"] = self.qf2.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path)
        for k in loaded_dict:
            if k not in ["_actor", "_qf1", "_qf2"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.actor.load_state_dict(loaded_dict["_actor"])
        model.qf1.load_state_dict(loaded_dict["_qf1"])
        model.qf2.load_state_dict(loaded_dict["_qf2"])
        model.qf1_target.load_state_dict(loaded_dict["_qf1"])
        model.qf2_target.load_state_dict(loaded_dict["_qf2"])
        return model

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env
