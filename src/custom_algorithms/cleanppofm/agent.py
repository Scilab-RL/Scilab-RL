import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.categorical import Categorical
from stable_baselines3.common.logger import Logger
from custom_algorithms.cleanppofm.utils import flatten_obs, layer_init, \
    get_position_and_object_positions_of_observation, get_observation_of_position_and_object_positions, \
    get_next_position_observation_moonlander
from custom_envs.moonlander.helper_functions import calculate_gaussian_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    """
    Agent class for the PPO algorithm. The agent has a critic and an actor network. It only works with discrete actions.
    """

    def __init__(self, env, reward_predicting: bool, model_based: bool = False) -> None:
        """
        Initialize the agent
        Args:
            env: using env for the observation space and action space size
        """
        super().__init__()
        self.model_based = model_based
        self.number_of_actions = env.action_space.n
        self.reward_predicting = reward_predicting

        # this is implemented for the gridworld envs
        if isinstance(env.observation_space, spaces.Dict):
            obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()])
            self.flatten = True
        # this is implemented for the moonlander env?
        else:
            obs_shape = np.array(env.observation_space.shape).prod()
            self.flatten = False
        if model_based:
            # actual obs + 3 predicted obs + 3 predicted rewards
            obs_shape = obs_shape * 4 + 3

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
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.action_space.n, device=device))
        self.env = env

    def get_value(self, obs) -> torch.Tensor:
        """
        Get the value of the critic network
        Args:
            obs: observation

        Returns: value of the critic network in a tensor

        """
        if self.flatten:
            obs = flatten_obs(obs)
        else:
            obs = torch.tensor(obs, device=device, dtype=torch.float32).detach().clone()
        return self.critic(obs)

    def get_action_and_value_and_forward_model_prediction(self, fm_network, obs, action=None,
                                                          deterministic: bool = False,
                                                          logger: Logger = None,
                                                          position_predicting: bool = False,
                                                          maximum_number_of_objects: int = 5) -> \
            tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Normal]:
        """
        Get the action, logarithmic probability of action distribution, entropy of the action distribution
        value of the critic network and forward model prediction in form of a normal distribution
        Args:
            fm_network: instance of the forward model network
            obs: observation
            action: optional action to take
            deterministic: if true the action selection is deterministic and not stochastic
            logger: logger for logging
            position_predicting: if true the forward model predicts the position of the agent out of the observation
            maximum_number_of_objects: maximal number of objects considered in the forward model prediction

        Returns:
            action, logarithmic probability of action distribution, entropy of the action distribution, value of the
             critic network and forward model prediction in form of a normal distribution
        """

        ##### FLATTEN OBS #####
        # flatten the obs when it is a dict
        if self.flatten:
            obs = flatten_obs(obs)
        else:
            if isinstance(obs, torch.Tensor):
                obs = obs.clone().detach()
            else:
                obs = torch.tensor(obs, device=device, dtype=torch.float32).clone().detach()

        observation_width = self.env.env_method("get_wrapper_attr", "observation_width")[0]
        observation_height = self.env.env_method("get_wrapper_attr", "observation_height")[0]
        agent_size = self.env.env_method("get_wrapper_attr", "size")[0]
        task = self.env.env_method("get_wrapper_attr", "task")[0]
        obs_for_agent = obs.clone().detach()

        if self.model_based:
            if not position_predicting:
                raise ValueError("Model based agent needs position predicting")

            action_to_prediction_dict = {}
            # Use fm
            comb_obs = obs.clone().detach()
            comb_obj_positions = get_position_and_object_positions_of_observation(comb_obs,
                                                                                  maximum_number_of_objects=maximum_number_of_objects,
                                                                                  observation_width=observation_width,
                                                                                  observation_height=observation_height,
                                                                                  agent_size=agent_size)

            # Convert comb_obj_positions to a tensor
            cop_tensor = torch.tensor(comb_obj_positions).float()

            obs_after_every_action = comb_obs.clone().detach()
            rewards_for_every_action = {}
            if task == "dodge":
                task_type = "obstacle"
            elif task == "collect":
                task_type = "coin"

            for i in range(0, self.number_of_actions):
                # Fixme: for hardcoded next obs, we had to change the ordering
                current_action = torch.full((cop_tensor.shape[0], 1), i)
                current_action_hardcoded = torch.full((1, cop_tensor.shape[0]), i)
                network_result_action = fm_network(cop_tensor, current_action)
                action_to_prediction_dict[str(i)] = network_result_action
                nra_mean = network_result_action.mean
                if self.reward_predicting:
                    nra_mean = nra_mean[:, :-1]

                # FIXME: hardcoded next obs
                next_positions = get_next_position_observation_moonlander(
                    observations=cop_tensor,
                    actions=current_action_hardcoded[0],
                    observation_width=observation_width,
                    observation_height=observation_height,
                    agent_size=agent_size,
                    maximum_number_of_objects=maximum_number_of_objects)

                obs_after_action = get_observation_of_position_and_object_positions(
                    # agent_and_object_positions=nra_mean,
                    agent_and_object_positions=next_positions,
                    observation_height=observation_height,
                    observation_width=observation_width,
                    agent_size=agent_size, task=task)

                # calculate reward for new obs
                x_position_of_agent = int(
                    min(max(agent_size, next_positions[0][0]), observation_width - agent_size + 1))
                y_position_of_agent = int(next_positions[0][1])

                collected_objects = []
                for index in range(2, len(next_positions[0]), 2):
                    if not (next_positions[0][index] == 0 and next_positions[0][index + 1] == 0):

                        if (
                                (
                                        ((next_positions[0][index] - 1) == (x_position_of_agent - 1))
                                        or ((next_positions[0][index] - 1) == x_position_of_agent)
                                        or ((next_positions[0][index] - 1) == (x_position_of_agent + 1))
                                        or (next_positions[0][index] == (x_position_of_agent - 1))
                                        or (next_positions[0][index] == x_position_of_agent)
                                        or (next_positions[0][index] == (x_position_of_agent + 1))
                                        or ((next_positions[0][index] + 1) == (x_position_of_agent - 1))
                                        or ((next_positions[0][index] + 1) == x_position_of_agent)
                                        or ((next_positions[0][index] + 1) == (x_position_of_agent + 1))
                                )
                                and
                                (
                                        ((next_positions[0][index + 1] - 1) == (y_position_of_agent - 1))
                                        or ((next_positions[0][index + 1] - 1) == y_position_of_agent)
                                        or ((next_positions[0][index + 1] - 1) == (y_position_of_agent + 1))
                                        or (next_positions[0][index + 1] == (y_position_of_agent - 1))
                                        or (next_positions[0][index + 1] == y_position_of_agent)
                                        or (next_positions[0][index + 1] == (y_position_of_agent + 1))
                                        or ((next_positions[0][index + 1] + 1) == (y_position_of_agent - 1))
                                        or ((next_positions[0][index + 1] + 1) == y_position_of_agent)
                                        or ((next_positions[0][index + 1] + 1) == (y_position_of_agent + 1))
                                )
                        ):
                            collected_objects.append(
                                {'x': int(next_positions[0][index]),
                                 'y': int(next_positions[0][index + 1]),
                                 'size': agent_size})

                # possibly a batch of 64, so call the function for each observation
                rewards_for_every_action[i] = [calculate_gaussian_reward(
                    state=np.array(row).reshape(observation_height, observation_width + 2),
                    collected_objects=collected_objects,
                    agent_size=agent_size,
                    task_type=task_type,
                    current_reward_function="gaussian",
                    x_position_of_agent=x_position_of_agent,
                    y_position_of_agent=y_position_of_agent)[0] for row in obs_after_action]
                obs_after_every_action = torch.cat((obs_after_every_action, obs_after_action), dim=1)

            for i in range(0, self.number_of_actions):
                rewards_for_every_action_tensor = torch.tensor(rewards_for_every_action[i]).reshape(
                    obs_after_every_action.shape[0],
                    1)
                obs_after_every_action = torch.cat((obs_after_every_action, rewards_for_every_action_tensor), dim=1)
            obs_for_agent = obs_after_every_action

        ##### PREDICT NEXT ACTION #####
        # obs is a tensor
        # get the mean of each action (discrete) from the actor network
        # e.g. a tensor of size (1, 8) for each of the 8 discrete actions in the gridworld envs
        action_mean = self.actor_mean(obs_for_agent)

        # create a categorical distribution from the mean of the actions
        # e.g. a tensor of size (1, 8) for the probability of each of 8 discrete actions in the gridworld envs
        # the probabilities equal to 1
        # From Wikipedia, the free encyclopedia:
        # In probability theory and statistics, a categorical distribution (also called a generalized Bernoulli
        # distribution, multinoulli distribution[1]) is a discrete probability distribution that describes the possible
        # results of a random variable that can take on one of K possible categories, with the probability of each
        # category separately specified. There is no innate underlying ordering of these outcomes, but numerical labels
        # are often attached for convenience in describing the distribution, (e.g. 1 to K). The K-dimensional
        # categorical distribution is the most general distribution over a K-way event; any other discrete distribution
        # over a size-K sample space is a special case. The parameters specifying the probabilities of each possible
        # outcome are constrained only by the fact that each must be in the range 0 to 1, and all must sum to 1.
        distribution = Categorical(logits=action_mean)

        # sample an action from the distribution if action is not none
        if action is None:
            if deterministic:
                action = torch.argmax(action_mean)
                forward_normal_action = action.unsqueeze(0).unsqueeze(0)
            else:
                action = distribution.sample()
                forward_normal_action = action.unsqueeze(0)
        else:
            forward_normal_action = action.unsqueeze(1)

        ##### PREDICT NEXT STATE #####
        # predict next state from last state and selected action
        # formal_normal_action in form of tensor([[action]])
        # get position of last state out of the observation --> moonlander specific implementation
        if position_predicting:
            # don't predict again and use the prediction above
            # fixme: doesn't work with batchsize bigger than 1
            # if self.model_based:
            #     forward_model_prediction_normal_distribution = action_to_prediction_dict[str(action.item())]
            # else:
            positions = get_position_and_object_positions_of_observation(obs,
                                                                         maximum_number_of_objects=maximum_number_of_objects,
                                                                         observation_width=observation_width,
                                                                         observation_height=observation_height,
                                                                         agent_size=agent_size)
            forward_model_prediction_normal_distribution = fm_network(positions, forward_normal_action.float())
        else:
            forward_model_prediction_normal_distribution = fm_network(obs, forward_normal_action.float())

        # TODO: put prediction of fm network into observation? --> standard deviation or whole observation?
        ##### LOGGING #####
        # std describes the (un-)certainty of the prediction of each pixel
        # more logging possible
        logger.record_mean("fm/stddev", forward_model_prediction_normal_distribution.stddev.mean().item())

        ##### RETURN #####
        # return predicted action, logarithmic probability of action distribution (one value in tensor),
        #  WIKIPEDIA Entropy: die durchschnittliche Anzahl von Entscheidungen (bits), die benötigt werden,
        #  um ein Zeichen aus einer Zeichenmenge zu identifizieren oder zu isolieren,
        #  anders gesagt: ein Maß, welches für eine Nachrichtenquelle den mittleren Informationsgehalt ausgegebener
        #  Nachrichten angibt.
        #  - sum of probabilities multiplied with the log_2 of the probabilities (one value in tensor)
        #  EXPLANATION ON REDDIT:
        #  It's a measure of how sure your classifier thinks it is.
        #  It maps the output vector between 0 (perfectly confident classification) and -ln(1/N),
        #  a large positive number (perfectly clueless classification).
        #  If your classifier puts all the weight onto one class, then it has minimal entropy
        #  (it believes it knows the exact class). If your classifier is totally unsure which class the example
        #  belongs to (every class' probability is equal), the entropy is maximal.
        #  There's not really a right answer to "what you want".
        #  You want the classifier to be right and sure in clear examples (so both correct and minimal entropy),
        #  but if you feed your cat vs dog classifier a picture containing both a cat and a dog,
        #  you probably want the model to give you a 50-50 split which is maximal entropy.
        # value of critic network, forward model prediction in normal distribution

        return action.unsqueeze(0), distribution.log_prob(action), distribution.entropy(), self.critic(
            obs_for_agent), forward_model_prediction_normal_distribution
