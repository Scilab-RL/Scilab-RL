# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from stable_baselines3.common.type_aliases import GymEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def flatten_obs(obs):
    observation, ag, dg = obs['observation'], obs['achieved_goal'], obs['desired_goal']
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).to(device)
    if isinstance(ag, np.ndarray):
        ag = torch.from_numpy(ag).to(device)
    if isinstance(dg, np.ndarray):
        dg = torch.from_numpy(dg).to(device)
    return torch.cat([observation, ag, dg], dim=1).to(dtype=torch.float32).detach().clone()


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
        else:
            obs_shape = np.array(env.observation_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class CLEANPPO:
    """
    Custom version of PPO adapted from CleanRL
    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
    https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

    :param env: a gymnasium environment with discrete action space
    :param learning_rate: the learning rate for the agent
    :param num_steps: the number of steps to run in each environment per policy rollout
    :param num_envs: the number of parallel environments
    :param anneal_lr: whether to use learning rate annealing for policy and value networks
    :param gamma: the discount factor gamma
    :param gae_lambda: the lambda for the general advantage estimation
    :param num_minibatches: the number of mini-batches
    :param update_epochs: the K epochs to update the policy
    :param clip_coef: the surrogate clipping coefficient
    :param norm_adv: whether to use advantages normalization
    :param clip_vloss:  whether to use a clipped loss for the value function, as per the paper
    :param ent_coef: coefficient of the entropy
    :param vf_coef: coefficient of the value function
    :param max_grad_norm: the maximum norm for the gradient clipping
    :param target_kl: the target KL divergence threshold
    """
    def __init__(self,
                 env: GymEnv,
                 learning_rate: float,
                 num_steps: int,
                 num_envs: int,
                 anneal_lr: bool,
                 gamma: float,
                 gae_lambda: float,
                 num_minibatches: int,
                 update_epochs: int,
                 clip_coef: float,
                 norm_adv: bool,
                 clip_vloss: bool,
                 ent_coef: float,
                 vf_coef: float,
                 max_grad_norm: float,
                 target_kl: float):
        self.env = env
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.dict_obs = True
            self.obs_shape = (np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()]),)
        else:
            self.dict_obs = False
            self.obs_shape = self.env.observation_space.shape
        assert isinstance(self.env.action_space, gym.spaces.Box), "only continuous action space is supported"

        self.agent = Agent(self.env).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // num_minibatches)

        # Storage setup
        self.obs = torch.zeros((num_steps, num_envs) + self.obs_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + self.env.action_space.shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

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

        global_step = 0
        next_obs = self.env.reset()
        if self.dict_obs:
            next_obs = flatten_obs(next_obs)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(self.num_envs).to(device)
        num_updates = total_timesteps // self.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                self.num_timesteps += 1 * self.num_envs  # this might skip some evaluations if self.num_envs > 1
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, infos = self.env.step(action.cpu().numpy())
                if self.dict_obs:
                    next_obs = flatten_obs(next_obs)
                self.rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if callback.on_step() is False:
                    return

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.obs_shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.logger.record("train/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.logger.record("train/value_loss", v_loss.item(), global_step)
            self.logger.record("train/policy_loss", pg_loss.item(), global_step)
            self.logger.record("train/entropy", entropy_loss.item(), global_step)
            self.logger.record("train/old_approx_kl", old_approx_kl.item(), global_step)
            self.logger.record("train/approx_kl", approx_kl.item(), global_step)
            self.logger.record("train/clip_fraction", np.mean(clipfracs), global_step)
            self.logger.record("train/explained_variance", explained_var, global_step)

    def predict(self, obs, state, episode_start, deterministic):
        with torch.no_grad():
            if self.dict_obs:
                obs = flatten_obs(obs)
            obs = torch.Tensor(obs).to(device)
            action, _, _, _ = self.agent.get_action_and_value(obs, deterministic=deterministic)
            action = action.cpu().numpy()
        return action, state

    def save(self, path):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "n_updates", "callback", "agent",
                           "obs", "actions", "logprobs", "rewards", "dones", "values"]:
            del data[to_exclude]
        # save network parameters
        data["_agent"] = self.agent.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path)
        for k in loaded_dict:
            if k not in ["_agent"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.agent.load_state_dict(loaded_dict["_agent"])
        return model

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env
