"""
This is an implementation of the One-step Actor-Critic (episodic) algorithm from the book
"Reinforcement Learning - An Introduction" by Richard S. Sutton and Andrew G. Barto, 2nd edition.
It is based on this pseudocode on page 332:

### One-step Actor–Critic (episodic), for estimating π(a|s,θ) ###

Input: a differentiable policy parameterization π(a|s,θ)
Input: a differentiable state-value function parameterization v̂(s,w)
Parameters: step sizes alpha_θ > 0, alpha_w > 0
Initialize policy parameter and state-value weights (e.g. to 0)
Loop forever (for each episode):
    Initialize S (first state of episode)
    I <- 1
    Loop while S is not terminal (for each time step):
        A ~ π(·|S,θ)
        Take action A, observe S', R
        δ <- R + γ * v̂(S',w) - v̂(S,w)         (if S' is terminal, then v̂(S',w) = 0)
        w <- w + alpha_w * δ * ∇ v̂(S,w)
        θ <- θ + alpha_θ * I * δ * ln ∇ π(A|S,θ)
        I <- γI
        S <- S'
"""
import torch
import torch.nn as nn
from gymnasium import spaces


class ONESTEPAC:
    """
    One-step Actor–Critic (episodic), for estimating π(a|s,θ)

    This algorithm follows the pseudocode above and has the pseudocode lines
    as comments throughout the implementation. They are marked with three #
    ### like this, this is a line from the pseudocode
    """

    def __init__(self, env):
        self.env = env
        self.num_timesteps = 0
        self.logger = None

        self.observation_space = spaces.flatten_space(env.observation_space)

        ### Input: a differentiable policy parameterization π(a|s,θ)
        # implemented as a neural network
        self.policy = nn.Sequential(nn.Linear(self.observation_space.shape[0], 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.env.action_space.shape[0]),
                                    nn.Softmax())

        ### Input: a differentiable state-value function parameterization v̂(s,w)
        # implemented as a neural network
        self.state_value_func = nn.Sequential(nn.Linear(self.observation_space.shape[0], 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 1))

        ### Parameters: step sizes alpha_θ > 0, alpha_w > 0
        # this is the learning-rate and we set alpha_θ == alpha_w
        self.learning_rate = 0.0003

        self.policy_optim = torch.optim.SGD(self.policy.parameters(), lr=self.learning_rate)
        self.state_value_func_optim = torch.optim.SGD(self.state_value_func.parameters(), lr=self.learning_rate)

        ### Initialize policy parameter and state-value weights (e.g. to 0)
        # no need because pytorch already handles the initialization

        # set γ
        self.gamma = 0.95

    def learn(self, total_timesteps, callback, log_interval):
        """
        The learning loop.
        The agent carries out a random action and receives an observation and a reward.
        """
        callback.init_callback(model=self)

        ### Initialize S (first state of episode)
        obs = self.env.reset()
        # preprocess the observation
        obs = torch.tensor(spaces.flatten(self.env.observation_space, obs), dtype=torch.float32)

        ### I <- 1
        I = 1

        criterion = nn.L1Loss()

        ### Loop forever (for each episode):
        ###     ...
        ###     Loop while S is not terminal (for each time step):
        # instead of looping over the episodes and timesteps, we loop over the total_timesteps,
        # which are computed like total_timesteps = n_episodes * n_steps_per_episode
        while self.num_timesteps < total_timesteps:

            ### A ~ π(·|S,θ)
            action = self.policy(obs)

            ### Take action A, observe S', R.
            # S' is "next_obs". R is "rewards"
            next_obs, rewards, dones, infos = self.env.step([action.detach().numpy()])
            next_obs = torch.tensor(spaces.flatten(self.env.observation_space, next_obs), dtype=torch.float32)
            reward = rewards[0]
            done = dones[0]

            ### δ <- R + γ * v̂(S',w) - v̂(S,w)         (if S' is terminal, then v̂(S',w) = 0)
            # we call v̂(S',w) "next_state_value" and v̂(S,w) "state_value"

            if done:  # if S' is terminal
                next_state_value = torch.zeros(1)  # then v̂(S',w) = 0
                callback.on_rollout_start()

                next_obs = self.env.reset()
                next_obs = torch.tensor(spaces.flatten(self.env.observation_space, next_obs), dtype=torch.float32)
                I = 1
            else:
                next_state_value = self.state_value_func(next_obs)  # v̂(S',w)

            state_value = self.state_value_func(obs)  # v̂(S',w)

            # delta = reward + self.gamma * next_state_value - state_value  # δ <- R + γ * v̂(S',w) - v̂(S,w)
            delta = criterion(reward + self.gamma * next_state_value, state_value)

            ### w <- w + alpha_w * δ * ∇ v̂(S,w)
            ### θ <- θ + alpha_θ * I * δ * ln ∇ π(A|S,θ)
            # instead of updating the parameters w and θ manually, we let the pytorch optimizer do the work
            self.policy_optim.zero_grad()
            self.state_value_func_optim.zero_grad()
            delta.backward()
            self.policy_optim.step()
            self.state_value_func_optim.step()

            ### I <- γI todo ignored for now

            if not callback.on_step():
                return

            self.num_timesteps += 1

            ### S <- S'
            obs = next_obs

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
        obs = torch.tensor(spaces.flatten(self.env.observation_space, obs), dtype=torch.float32)
        return [self.policy(obs).detach().numpy()], state

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env
