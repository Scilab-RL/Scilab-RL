import numpy as np
import torch
import math
import copy

from custom_envs.moonlander.helper_functions import calculate_gaussian_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_obs(obs: dict) -> torch.Tensor:
    """
    Flatten a dict observation of the Gridworld envs.
    Args:
        obs: observation of type dict

    Returns:
        flattened observation in tensor format

    """
    # tensor can not check for string ("agent" in obs)
    if isinstance(obs, dict):
        # this is the flatten process for the Gridworld envs
        agent, target = obs['agent'], obs['target']
        if isinstance(agent, np.ndarray):
            agent = torch.from_numpy(agent).to(device)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).to(device)
        return torch.cat([agent, target], dim=1).to(dtype=torch.float32).detach().clone()
    else:
        raise NotImplementedError(
            "Flatten observation not implemented for this environment with this observation type.")


def layer_init(layer, std: np.float64 = np.sqrt(2), bias_const: float = 0.0) -> torch.nn.Module:
    """
    Initialize the layer with a (semi) orthogonal matrix and a constant bias.
    Args:
        layer: layer to be initialized
        std: standard deviation for the orthogonal matrix
        bias_const: constant for the bias

    Returns:
        initialized layer

    """
    # Fill the layer weight with a (semi) orthogonal matrix.
    # Described in Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    # Saxe, A. et al. (2013).
    torch.nn.init.orthogonal_(layer.weight, std)
    # Fill the layer bias with the value bias_const.
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_reward_estimation_of_forward_model(fm_network, obs: torch.Tensor,
                                           position_predicting: bool,
                                           default_action: torch.Tensor = torch.Tensor([[1]]),
                                           number_of_future_steps: int = 10,
                                           maximum_number_of_objects: int = 5) -> float:
    """
    Get the reward estimation of the forward model for number_of_future_steps steps.
    Args:
        fm_network: forward model network
        obs: last actual observation
        position_predicting: boolean if the forward model is predicting the position or actual observation
        default_action: action that is executed when not controlling the environment
        number_of_future_steps: number of future steps to predict
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction

    Returns:
        reward estimation of the forward model for number_of_future_steps steps

    """
    # the first obs is a matrix, the following obs are just the positions
    if position_predicting:
        # FIXME: missing observation height and width and agent size -> currently not used
        obs = get_position_and_object_positions_of_observation(obs,
                                                               maximum_number_of_objects=maximum_number_of_objects)

    ##### PREDICT NEXT REWARDS #####
    # predict next rewards from last state and default action
    reward_estimation = 0
    for i in range(number_of_future_steps):
        forward_model_prediction_normal_distribution = fm_network(obs, default_action.float())
        # add reward estimation to last reward
        reward_estimation += forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][-1]
        ##### REMOVE REWARD FROM NEW PREDICTED OBS #####
        obs = torch.clamp(torch.round(forward_model_prediction_normal_distribution.mean[0][:-1].unsqueeze(dim=0)),
                          min=0,
                          max=4)
    return reward_estimation


def get_summed_up_reward_of_env_or_fm_with_predicted_states_of_fm(env, fm_network, last_observation: torch.Tensor,
                                                                  reward_from_env: bool,
                                                                  position_predicting: bool, env_name: str,
                                                                  number_of_future_steps: int = 10,
                                                                  maximum_number_of_objects: int = 5) -> float:
    """
    Get the reward of the forward model prediction or environment through predicted states of the forward model
     for number_of_future_steps steps.
    Args:
        env: environment
        fm_network: forward model network
        last_observation: last known observation in form of positions
        reward_from_env: boolean if the reward is calculated from the environment or the forward model
        position_predicting: boolean if the forward model is predicting the position or actual observation
        env_name: name of the current environment
        number_of_future_steps: number of future steps to predict
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction

    Returns:
        summed up reward of the forward model or environment for number_of_future_steps steps

    """
    # default action is stay at same position
    if env_name == "MoonlanderWorldEnv":
        default_action = torch.tensor([[1]]).to(device)
    # using reward model of other envs is not implemented by now
    else:
        raise ValueError(
            f"The current environment does not support the reward calculation for {number_of_future_steps} steps.")
    task = env.env_method("get_wrapper_attr", "task")[0]
    if task == "dodge":
        task_type = "obstacle"
    elif task == "collect":
        task_type = "coin"

    observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
    observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
    agent_size = env.env_method("get_wrapper_attr", "size")[0]

    summed_up_reward = 0
    for i in range(number_of_future_steps):
        # if not done:
        # we manually predict the next state
        # positions for forward model
        if not position_predicting:
            # FIXME: not tested
            last_observation = torch.tensor(last_observation, device=device,
                                            dtype=torch.float32).detach().clone()
        forward_model_prediction_normal_distribution = fm_network(last_observation, default_action)

        # get reward from forward model prediction or environment
        if not reward_from_env:
            rewards = np.expand_dims(forward_model_prediction_normal_distribution.mean[0][
                                         -1].cpu().detach().numpy(), axis=0)
            # remove reward from new predicted obs
            last_observation = torch.clamp(
                torch.round(forward_model_prediction_normal_distribution.mean[0][:-1].unsqueeze(dim=0)),
                min=0,
                # FIXME: why 4?
                max=4)
        else:
            # define state for env
            # FIXME: only when reward is predicted by the forward model
            # FIXME: get last observation hardcoded:
            # last_observation = forward_model_prediction_normal_distribution.mean[0][:-1].cpu().unsqueeze(0)
            last_observation = get_next_position_observation_moonlander(
                observations=last_observation,
                actions=default_action[0],
                observation_width=observation_width,
                observation_height=observation_height,
                agent_size=agent_size,
                maximum_number_of_objects=maximum_number_of_objects)
            last_observation_state = np.expand_dims(
                get_observation_of_position_and_object_positions(agent_and_object_positions=last_observation,
                                                                 observation_height=observation_height,
                                                                 observation_width=observation_width,
                                                                 agent_size=agent_size,
                                                                 task=task).flatten().cpu().numpy(),
                axis=0)

            x_position_of_agent = int(min(max(agent_size, last_observation[0][0]), observation_width - agent_size + 1))
            y_position_of_agent = int(last_observation[0][1])
            collected_objects = []
            for index in range(2, len(last_observation[0]), 2):
                if (((last_observation[0][index] == x_position_of_agent - 1)
                     or (last_observation[0][index] == x_position_of_agent)
                     or (last_observation[0][index] == x_position_of_agent + 1))
                        and ((last_observation[0][index + 1] == y_position_of_agent - 1)
                             or (last_observation[0][index + 1] == y_position_of_agent)
                             or (last_observation[0][index + 1] == y_position_of_agent + 1))
                ):
                    collected_objects.append(
                        {'x': int(last_observation[0][index]), 'y': int(last_observation[0][index + 1]),
                         'size': agent_size})
            rewards, _ = calculate_gaussian_reward(
                state=last_observation_state.reshape(observation_height, observation_width + 2),
                collected_objects=collected_objects,
                agent_size=agent_size,
                task_type=task_type,
                current_reward_function="gaussian",
                x_position_of_agent=x_position_of_agent,
                y_position_of_agent=y_position_of_agent)

        # normalize reward
        normalized_reward = normalize_rewards(task=task, absolute_reward=rewards)
        summed_up_reward += normalized_reward

    # normalize with mean of summed_up_reward
    return summed_up_reward / number_of_future_steps


def get_reward_with_future_reward_estimation_corrective(rewards: torch.Tensor, future_reward_estimation: float,
                                                        prediction_error: float) -> torch.Tensor:
    """
    Get the reward with future reward estimation corrective.
    Args:
        rewards: rewards of last step
        future_reward_estimation: estimation of future rewards
        prediction_error: error between predicted and last actual observation

    Returns:
        reward with future reward estimation corrective

    """
    rewards_with_future_reward_estimation = rewards + future_reward_estimation
    # different reward estimation for positive and negative rewards
    if rewards_with_future_reward_estimation < 0:
        # negative rewards
        reward_with_future_reward_estimation_corrective = rewards_with_future_reward_estimation + (
                abs(rewards_with_future_reward_estimation) * (1 - prediction_error))
    else:
        # positive rewards
        if not prediction_error == 0:
            reward_with_future_reward_estimation_corrective = rewards_with_future_reward_estimation / prediction_error
        # prediction error is zero -> we cannot divide by zero
        # -> give a boost of * 10 = perfect prediction (* 10 is similar range than other rewards)
        else:
            reward_with_future_reward_estimation_corrective = rewards_with_future_reward_estimation * 10
    return reward_with_future_reward_estimation_corrective


def reward_estimation(fm_network, new_obs: np.array, env_name: str, rewards, prediction_error: float,
                      position_predicting: bool, number_of_future_steps: int = 10, maximum_number_of_objects: int = 5):
    # default action is stay at same position
    if env_name == "MoonlanderWorldEnv":
        default_action = torch.tensor([[1]], device=device)
    # random default action for gridworld env
    elif env_name == "GridWorldEnv":
        default_action = torch.randint(low=0, high=8, size=(1, 1), device=device)
    else:
        raise ValueError("Environment not supported")

    # TODO: do this only after a warm-up phase of the forward model
    ##### REWARD ESTIMATION #####
    future_reward_estimation = get_reward_estimation_of_forward_model(
        fm_network=fm_network,
        obs=new_obs,
        position_predicting=position_predicting,
        default_action=default_action,
        number_of_future_steps=number_of_future_steps, maximum_number_of_objects=maximum_number_of_objects)
    reward_with_future_reward_estimation_corrective = get_reward_with_future_reward_estimation_corrective(
        rewards=rewards, future_reward_estimation=future_reward_estimation,
        prediction_error=prediction_error)

    return reward_with_future_reward_estimation_corrective


def get_position_and_object_positions_of_observation(obs: torch.Tensor,
                                                     # FIXME: sometimes the default value is used and we cannot use the value defined in the yaml file
                                                     # quickfix: change here the number to 5 (for small environments) and to 10 (for human environments)
                                                     maximum_number_of_objects: int = 10,
                                                     observation_width: int = 10,
                                                     observation_height: int = 10,
                                                     agent_size: int = 1) -> torch.Tensor:
    """
    Get the position of the agent and up to maximum_number_of_objects objects in the observation.
    Args:
        obs: observation
        maximum_number_of_objects: the number of objects that are considered in the observation
        observation_width: width of the observation
        agent_size: size of the agent in the observation
        # FIXME: note, these are the first objects you get, when going down in the obs and going from left to right
        # FIXME: these are not necessary the nearest objects to the agent

    Returns:
        position of the agent and maximum_number_of_objects objects in the observation, where the first two elements
        are the x and y position of the agent
    """
    agent_and_object_positions = []
    for obs_element in obs:
        current_number_of_objects_in_list = 0
        # agent in observation is marked with 1
        first_index_with_one = np.where(obs_element.cpu() == 1)[0][0]

        # object in observation is marked with 2 or 3
        if 2 in obs_element or 3 in obs_element:
            if 2 in obs_element:
                search_value = 2
            else:
                search_value = 3
            x_y_coordinates = []

            if agent_size == 1:
                indices_with_two_or_three = np.where(obs_element.cpu() == search_value)[0]
            elif agent_size == 2:
                # Find indices where three consecutive ones occur
                indices_with_two_or_three = np.where(
                    ((obs_element.cpu()[:-2] == search_value) | (obs_element.cpu()[:-2] == 1))
                    & ((obs_element.cpu()[1:-1] == search_value) | (obs_element.cpu()[1:-1] == 1))
                    & ((obs_element.cpu()[2:] == search_value) | (obs_element.cpu()[2:] == 1))
                )[0]
                mask = (
                    # check if there is a line above
                        (np.isin(indices_with_two_or_three - (observation_width + 2), indices_with_two_or_three)
                         # and check if there is a line below
                         & np.isin(indices_with_two_or_three + (observation_width + 2), indices_with_two_or_three))
                        # otherwise check if the object is flying out of the grid (index in first line & not an object line two rows below)
                        | ((indices_with_two_or_three < observation_width + 2)
                           & (np.isin(indices_with_two_or_three + ((observation_width + 2) * 2),
                                      indices_with_two_or_three,
                                      invert=True)))
                        # otherwise check if the object is flying into the grid (index in last line) & not two lines above (then it is covered above)
                        | ((indices_with_two_or_three >= (observation_width + 2) * (observation_height - 1))
                           & (np.isin(indices_with_two_or_three - ((observation_width + 2) * 2),
                                      indices_with_two_or_three, invert=True)))
                )
                indices_with_two_or_three = indices_with_two_or_three[mask]
            else:
                raise ValueError("Agent size not supported.")

            for index in indices_with_two_or_three:
                # break if we have enough objects
                if current_number_of_objects_in_list >= maximum_number_of_objects:
                    break
                # get x and y coordinate of object
                # +2 because of the walls
                x_coordinate = (index % (observation_width + 2)) + agent_size - 1
                # get to the middle of the object
                y_coordinate = math.floor(index / (observation_width + 2))
                # remove agent from indices with two or three --> agent is added later at the beginning of the list
                if not (x_coordinate == (first_index_with_one + agent_size - 1) and y_coordinate == (agent_size - 1)):
                    x_y_coordinates.append([x_coordinate, y_coordinate])
                    current_number_of_objects_in_list += 1

            x_y_coordinates_copy = copy.deepcopy(x_y_coordinates)
            for current_x_coordinate, current_y_coordinate in x_y_coordinates:
                # check if object coordinate is overlapping with the agent at the same y position
                if (
                        (
                                ((current_x_coordinate - 1) == (first_index_with_one + agent_size - 2)) or
                                ((current_x_coordinate - 1) == (first_index_with_one + agent_size - 1)) or
                                ((current_x_coordinate - 1) == (first_index_with_one + agent_size)) or
                                (current_x_coordinate == (first_index_with_one + agent_size - 2)) or
                                (current_x_coordinate == (first_index_with_one + agent_size - 1)) or
                                (current_x_coordinate == (first_index_with_one + agent_size)) or
                                ((current_x_coordinate + 1) == (first_index_with_one + agent_size - 2)) or
                                ((current_x_coordinate + 1) == (first_index_with_one + agent_size - 1)) or
                                ((current_x_coordinate + 1) == (first_index_with_one + agent_size))
                        )  # and
                        # (
                        #         current_y_coordinate == (agent_size - 1)
                        # )
                ):
                    # object is left from agent
                    if current_x_coordinate < (first_index_with_one + agent_size - 1):
                        # check if object already started earlier
                        if obs_element.cpu()[current_x_coordinate - 2] == search_value:
                            x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])
                            current_number_of_objects_in_list -= 1
                    elif current_x_coordinate > (first_index_with_one + agent_size - 1):
                        if obs_element.cpu()[current_x_coordinate + 2] == search_value:
                            x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])
                            current_number_of_objects_in_list -= 1

                # check if object coordinate is overlapping with the agent at the same x position
                if (
                        (
                                ((current_y_coordinate - 1) == (agent_size - 2)) or
                                ((current_y_coordinate - 1) == (agent_size - 1)) or
                                ((current_y_coordinate - 1) == agent_size) or
                                (current_y_coordinate == (agent_size - 2)) or
                                (current_y_coordinate == (agent_size - 1)) or
                                (current_y_coordinate == agent_size) or
                                ((current_y_coordinate + 1) == (agent_size - 2)) or
                                ((current_y_coordinate + 1) == (agent_size - 1)) or
                                ((current_y_coordinate + 1) == agent_size)
                        ) and
                        (
                                current_x_coordinate == (first_index_with_one + agent_size - 1)
                        )
                ):
                    # object is below agent
                    if obs_element.cpu()[
                        ((current_y_coordinate + 2) * (observation_width + 2)) + current_x_coordinate] == search_value:
                        x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])
                        current_number_of_objects_in_list -= 1

            x_y_coordinates = x_y_coordinates_copy
            # add zeros to the list if we have not enough objects
            while current_number_of_objects_in_list < maximum_number_of_objects:
                x_y_coordinates.append([0, 0])
                current_number_of_objects_in_list += 1

            # add agent to object positions
            x_y_coordinates = [[first_index_with_one + agent_size - 1, agent_size - 1]] + x_y_coordinates
            agent_and_object_positions.append(x_y_coordinates)
        else:  # no object in observation
            # add agent to object positions + zeros for objects
            x_y_coordinates = [[first_index_with_one + agent_size - 1, agent_size - 1]] + [[0, 0] for i in
                                                                                           range(
                                                                                               maximum_number_of_objects)]
            agent_and_object_positions.append(x_y_coordinates)

    agent_and_object_positions_tensor = torch.flatten(
        torch.tensor(agent_and_object_positions, device=device, dtype=torch.float32),
        start_dim=1)

    return agent_and_object_positions_tensor


def get_next_whole_observation(next_observations: torch.Tensor, actions: torch.Tensor, observation_width: int,
                               observation_height: int) -> torch.Tensor:
    """
    Calculate the next observation in the moonlander environment manually to exclude random observations through input noise.
    Args:
        next_observations: next observations
        actions: actions

    Returns:
        next observation in the moonlander environment without input noise

    """
    # object in observation is marked with 2 or 3
    if 2 in next_observations:
        search_value = 2
    else:
        search_value = 3

    # deep copy of next_observation
    next_observations_copy = next_observations.detach().clone()
    # remove agent and objects
    next_observations_copy[next_observations_copy == 1] = 0
    next_observations_copy[next_observations_copy == 2] = 0

    # get x coordinates of agent for each batch element
    x_index_of_agent = torch.nonzero(next_observations == 1, as_tuple=True)[1]
    # get x, y coordinates of objects for each batch element (index of batch element in element_in_batch)
    element_in_batch, indices_of_objects = torch.nonzero(next_observations == search_value, as_tuple=True)
    # get x and y coordinate of object
    x_coordinate_tensor = indices_of_objects % (observation_width + 2)
    y_coordinate_tensor = torch.floor(indices_of_objects / (observation_height + 2))

    new_y_coordinate_tensor = y_coordinate_tensor - 1
    new_indices_of_objects = ((new_y_coordinate_tensor * (observation_width + 2)) + x_coordinate_tensor).int()

    # remove negative y coordinates (object flew out of the grid)
    valid_mask = new_indices_of_objects >= 0
    valid_element_in_batch = element_in_batch[valid_mask]
    valid_new_indices_of_objects = new_indices_of_objects[valid_mask]

    # add objects to next observation
    next_observations_copy[valid_element_in_batch, valid_new_indices_of_objects] = 2

    # add agent to next observation
    new_x_index_of_agent = torch.clamp(x_index_of_agent + (actions - 1), min=1, max=observation_width)
    next_observations_copy[torch.arange(next_observations.shape[0]), new_x_index_of_agent] = 1

    return next_observations_copy


def get_observation_of_position_and_object_positions(agent_and_object_positions: torch.Tensor, observation_height: int,
                                                     observation_width: int, agent_size: int,
                                                     task: str) -> torch.Tensor:
    observations = []
    for agent_and_object_position in agent_and_object_positions:
        # tensor of (1,12) or (1, 1260)
        copy_of_agent_and_object_position = agent_and_object_position.clone().detach()
        # build empty obs
        matrix = np.zeros(shape=(observation_height, observation_width + 2), dtype=np.int16)

        if task == "dodge":
            object_value = 3
        elif task == "collect":
            object_value = 2
        else:
            raise ValueError("Task not supported.")

        # add objects
        counter = 2
        while counter < len(copy_of_agent_and_object_position):
            x_position_of_object = int(torch.round(copy_of_agent_and_object_position[counter]))
            if agent_size <= x_position_of_object <= observation_width + 1 - agent_size:
                matrix[
                max(0, min(observation_height - (2 * agent_size - 1),
                           int(torch.round(
                               copy_of_agent_and_object_position[
                                   counter + 1]) - agent_size + 1))):  # y start of object
                max(2 * agent_size - 2, min(observation_height - 1, int(torch.round(
                    copy_of_agent_and_object_position[counter + 1]) + agent_size - 1))) + 1,  # y end of object
                x_position_of_object - agent_size + 1:  # x start of object
                x_position_of_object + agent_size] = object_value  # x end of object
            counter += 2

        # add agent
        x_position_of_agent = int(torch.round(copy_of_agent_and_object_position[0]))
        # make sure that the agent is within the matrix
        if x_position_of_agent < agent_size:
            x_position_of_agent = agent_size
        elif x_position_of_agent > observation_width + 1 - agent_size:
            x_position_of_agent = observation_width + 1 - agent_size

        # first element is the y position of the agent, second element is the x position of the agent
        matrix[
        max(0, min(observation_height - (2 * agent_size - 1),
                   int(torch.round(copy_of_agent_and_object_position[1]) - agent_size + 1))):  # y start of agent
        max(2 * agent_size - 2,
            min(observation_height - 1,
                int(torch.round(copy_of_agent_and_object_position[1]) + agent_size - 1))) + 1,
        # y end of agent
        x_position_of_agent - agent_size + 1:  # x start of agent
        x_position_of_agent + agent_size] = 1  # x end of agent

        # add wall
        matrix[:, 0] = -1
        matrix[:, -1] = -1

        observations.append(matrix)

    observations_tensor = torch.flatten(torch.tensor(observations, device=device, dtype=torch.float32), start_dim=1)

    return observations_tensor


def get_next_observation_gridworld(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Calculate the next observation in the gridworld environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions

    Returns:
        next observation in the gridworld environment without input noise

    """
    ##### WHILE THE AGENT IS TRAINED WITH INPUT NOISE, THE FM IS TRAINED WITHOUT INPUT NOISE
    action_to_direction = {
        0: np.array([1, 0]),  # right
        1: np.array([1, 1]),  # right down (diagonal)
        2: np.array([0, 1]),  # down
        3: np.array([-1, 1]),  # left down (diagonal)
        4: np.array([-1, 0]),  # left
        5: np.array([-1, -1]),  # left up
        6: np.array([0, -1]),  # up
        7: np.array([1, -1])  # right up
    }
    agent_location_without_input_noise = torch.empty(size=(observations.shape[0], 4), device=device)
    for index, action in enumerate(actions):
        direction = action_to_direction[int(action)]
        # We use `np.clip` to make sure we don't leave the grid
        standard_agent_location = np.clip(
            np.array(observations[index][0:2].cpu()) + direction, 0, 4
        )
        agent_location_without_input_noise[index] = torch.tensor(
            np.concatenate((standard_agent_location, observations[index][2:4].cpu())), device=device
        )
    return agent_location_without_input_noise


def get_next_position_observation_moonlander(observations: torch.Tensor, actions: torch.Tensor, observation_width: int,
                                             observation_height: int, agent_size: int,
                                             maximum_number_of_objects: int) -> torch.Tensor:
    """
    Calculate the next observation in the moonlander environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions

    Returns:
        next observation in the moonlander environment without input noise
    """
    next_observation_without_input_noise = observations.clone().detach()

    for index, obs in enumerate(observations):
        # first two elements are agent x and y position
        counter = 2
        # index[0] is agent x position
        # apply action to agent x position (actions: 0, 1, 2)
        # action_movement is -1 to go left, 0 to stay and 1 to go right for agent size 1
        # for agent size 2 it is -2, 0, 2
        # for agent size 3 it is -3, 0, 3 ...
        next_observation_without_input_noise[index][0] += (agent_size * actions[index] - agent_size)
        # clip to range of agent_size to observation_width
        next_observation_without_input_noise[index][0] = torch.clamp(next_observation_without_input_noise[index][0],
                                                                     agent_size, observation_width - agent_size + 1)

        # check if there is an object that already is on position 0 -> removed
        while counter < (maximum_number_of_objects * 2):
            if not next_observation_without_input_noise[index][counter] == 0 and \
                    next_observation_without_input_noise[index][counter + 1] == 0:
                next_observation_without_input_noise[index][counter] = 0
                next_observation_without_input_noise[index][counter + 1] = 0
            counter += 2

        # apply step to every y position (agent and objects)
        next_observation_without_input_noise[index][1::2] -= 1
        # clip agent (cannot go out of the grid)
        next_observation_without_input_noise[index][1] = torch.clamp(
            next_observation_without_input_noise[index][1], agent_size - 1, observation_height - agent_size)
        # clip objects (can be at y position zero, when they go out of the grid)
        next_observation_without_input_noise[index][3::2] = torch.clamp(
            next_observation_without_input_noise[index][3::2], 0, observation_height - agent_size)

    return next_observation_without_input_noise


def calculate_prediction_error(env_name: str, env, next_obs, forward_model_prediction_normal_distribution: torch.normal,
                               maximum_number_of_objects: int = 5) -> float:
    """
    Calculate the prediction error between the next obs and the forward model prediction.
    Args:
        env_name: name of the environment
        env: environment
        next_obs: observation after actually executing the action
        forward_model_prediction_normal_distribution: prediction of next observation by forward model
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction

    Returns:
        prediction error between the next obs and the forward model prediction
    """
    observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
    observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
    agent_size = env.env_method("get_wrapper_attr", "size")[0]
    ##### CALCULATE PREDICTION ERROR #####
    # prediction error version one -> standard deviation
    # prediction_error = forward_normal.stddev.mean().item()
    # prediction error version two -> Euclidean distance
    # calculate manually prediction error (Euclidean distance)
    if env_name == "GridWorldEnv":
        # predicted location can only be between 0 and 4
        max_distance_in_gridworld = math.sqrt(((4 - 0) ** 2) + ((4 - 0) ** 2) + ((4 - 0) ** 2) + ((4 - 0) ** 2))
        predicted_location = torch.tensor(
            [min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][0])), 4),
             min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][1])), 4),
             min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][2])), 4),
             min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][3])), 4)],
            device=device)
        prediction_error = (math.sqrt(torch.sum((predicted_location - next_obs) ** 2))) / max_distance_in_gridworld
    elif env_name == "MoonlanderWorldEnv":
        # we just care for the x position of the moonlander agent, because the y position is always equally to the size of the agent
        # independently of using the whole obs or the position prediction, we use the position predictions to calculate the prediction error
        positions = get_position_and_object_positions_of_observation(next_obs,
                                                                     maximum_number_of_objects=maximum_number_of_objects,
                                                                     observation_width=observation_width,
                                                                     observation_height=observation_height,
                                                                     agent_size=agent_size)

        # Smallest x position of the agent is the size of the agent
        # Biggest x position of the agent is the width of the moonlander world - the size of the agent
        # Note: you should use vec_env.env_method("get_wrapper_attr", "attribute_name") in Gymnasium v1.0
        first_possible_x_position = env.env_method("get_wrapper_attr", "first_possible_x_position")[0]
        last_possible_x_position = env.env_method("get_wrapper_attr", "last_possible_x_position")[0]
        max_distance_in_moonlander_world = math.sqrt(
            (last_possible_x_position - first_possible_x_position) ** 2)
        predicted_x_position = torch.tensor([min(max(first_possible_x_position,
                                                     forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[
                                                         0][0]),
                                                 last_possible_x_position)], device=device)
        prediction_error = (math.sqrt(
            torch.sum((predicted_x_position - positions[0][0]) ** 2))) / max_distance_in_moonlander_world
    else:
        raise ValueError("Environment not supported")

    return prediction_error


def calculate_difficulty(env, policy, fm_network, logger, env_name: str,
                         prediction_error: float, position_predicting: bool, maximum_number_of_objects: int = 5,
                         reward_predicting: bool = False, use_reward_of_env: bool = False) -> tuple[float, float]:
    """
    Calculate the difficulty of the environment by simulating the default trajectory
    and the "optimal" trajectory the agent would choose.
    Args:
        env: environment
        policy: agent policy to predict actions
        fm_network: forward model network
        logger: logger
        env_name: name of the environment
        prediction_error: error between predicted and last actual observation
        position_predicting: if the forward model is predicting the position or actual observation
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction
        reward_predicting: if the forward model is predicting the reward or the environment
        use_reward_of_env: if the reward of the environment should be used
    Returns:
        difficulty between 0 and 1
        summed up rewards when executing the default action (trajectory length is calculated by prediction error)
    """
    # default action is stay at same position
    if env_name == "MoonlanderWorldEnv":
        default_action = torch.tensor([[1]]).to(device)
    # using reward model of other envs is not implemented by now
    else:
        raise ValueError(
            "The current environment does not support the difficulty calculation.")
    task = env.env_method("get_wrapper_attr", "task")[0]
    if task == "dodge":
        task_type = "obstacle"
    elif task == "collect":
        task_type = "coin"

    # calculate the trajectory lengths through the prediction error
    # we decide that the trajectory length is half the observation size of the environment when the prediction error is 0
    observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
    observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
    agent_size = env.env_method("get_wrapper_attr", "size")[0]
    trajectory_length = - (observation_height / 2) * prediction_error + observation_height / 2

    last_observation_state_default = np.expand_dims(env.env_method("get_wrapper_attr", "state")[0].flatten(), axis=0)
    last_observation_state_optimal = copy.deepcopy(last_observation_state_default)

    # simulate the default and optimal trajectory
    copied_env_default = copy.deepcopy(env)
    copied_env_optimal = copy.deepcopy(env)

    # remove possible input noise in the environment
    copied_env_default.env_method("set_input_noise", 0)
    copied_env_optimal.env_method("set_input_noise", 0)

    done_default = copied_env_default.env_method("is_done")[0]
    done_optimal = copied_env_optimal.env_method("is_done")[0]

    ##### CALCULATE NEXT REWARDS THROUGH ENVIRONMENT #####
    summed_up_reward_default = 0
    summed_up_reward_optimal = 0
    # simulate at least one step
    for i in range(max(round(trajectory_length), 1)):
        if not done_default:
            # we manually predict the next state
            # FIXME: don't want to use the forward model prediction here?
            # positions for forward model
            # if position_predicting:
            #     last_observation_default = get_position_and_object_positions_of_observation(
            #         torch.tensor(last_observation_default, device=device),
            #         maximum_number_of_objects=maximum_number_of_objects,
            #         observation_width=observation_width, observation_height=observation_height, agent_size=agent_size)
            # else:
            #     last_observation_default = torch.tensor(last_observation_default, device=device,
            #                                             dtype=torch.float32).detach().clone()

            # forward_model_prediction_normal_distribution_default = fm_network(last_observation_default, default_action)

            # get positions
            last_observation_default = get_position_and_object_positions_of_observation(
                torch.tensor(last_observation_state_default, device=device),
                maximum_number_of_objects=maximum_number_of_objects,
                observation_width=observation_width, observation_height=observation_height, agent_size=agent_size)
            # get next positions for new step with default action
            last_observation_default = get_next_position_observation_moonlander(
                observations=last_observation_default,
                actions=default_action[0],
                observation_width=observation_width,
                observation_height=observation_height,
                agent_size=agent_size,
                maximum_number_of_objects=maximum_number_of_objects)

            # get reward from forward model prediction or environment
            # if reward_predicting and not use_reward_of_env:
            #     # state for env
            #     last_observation_default = np.expand_dims(
            #         get_observation_of_position_and_object_positions(agent_and_object_positions=
            #         forward_model_prediction_normal_distribution_default.mean[0][:-1].cpu().unsqueeze(
            #             0), observation_height=observation_height,
            #             observation_width=observation_width, agent_size=agent_size, task=task).flatten().cpu().numpy(),
            #         axis=0)
            #     rewards_default = np.expand_dims(forward_model_prediction_normal_distribution_default.mean[0][
            #                                          -1].cpu().detach().numpy(), axis=0)
            # else:
            # _, rewards_default, done_default, _ = copied_env_default.step(default_action)

            x_position_of_agent = int(
                min(max(agent_size, last_observation_default[0][0]), observation_width - agent_size + 1))
            y_position_of_agent = int(last_observation_default[0][1])

            collected_objects = []
            for index in range(2, len(last_observation_default[0]), 2):
                if not (last_observation_default[0][index] == 0 and last_observation_default[0][index + 1] == 0):

                    if (
                            (
                                    ((last_observation_default[0][index] - 1) == (x_position_of_agent - 1))
                                    or ((last_observation_default[0][index] - 1) == x_position_of_agent)
                                    or ((last_observation_default[0][index] - 1) == (x_position_of_agent + 1))
                                    or (last_observation_default[0][index] == (x_position_of_agent - 1))
                                    or (last_observation_default[0][index] == x_position_of_agent)
                                    or (last_observation_default[0][index] == (x_position_of_agent + 1))
                                    or ((last_observation_default[0][index] + 1) == (x_position_of_agent - 1))
                                    or ((last_observation_default[0][index] + 1) == x_position_of_agent)
                                    or ((last_observation_default[0][index] + 1) == (x_position_of_agent + 1))
                            )
                            and
                            (
                                    ((last_observation_default[0][index + 1] - 1) == (y_position_of_agent - 1))
                                    or ((last_observation_default[0][index + 1] - 1) == y_position_of_agent)
                                    or ((last_observation_default[0][index + 1] - 1) == (y_position_of_agent + 1))
                                    or (last_observation_default[0][index + 1] == (y_position_of_agent - 1))
                                    or (last_observation_default[0][index + 1] == y_position_of_agent)
                                    or (last_observation_default[0][index + 1] == (y_position_of_agent + 1))
                                    or ((last_observation_default[0][index + 1] + 1) == (y_position_of_agent - 1))
                                    or ((last_observation_default[0][index + 1] + 1) == y_position_of_agent)
                                    or ((last_observation_default[0][index + 1] + 1) == (y_position_of_agent + 1))
                            )
                    ):
                        collected_objects.append(
                            {'x': int(last_observation_default[0][index]),
                             'y': int(last_observation_default[0][index + 1]),
                             'size': agent_size})

            # state for env
            last_observation_state_default = np.expand_dims(
                get_observation_of_position_and_object_positions(agent_and_object_positions=last_observation_default,
                                                                 observation_height=observation_height,
                                                                 observation_width=observation_width,
                                                                 agent_size=agent_size,
                                                                 task=task).flatten().cpu().numpy(),
                axis=0)
            # calculate reward
            rewards_default, _ = calculate_gaussian_reward(
                state=last_observation_state_default.reshape(observation_height, observation_width + 2),
                collected_objects=collected_objects,
                agent_size=agent_size,
                task_type=task_type,
                current_reward_function="gaussian",
                x_position_of_agent=x_position_of_agent,
                y_position_of_agent=y_position_of_agent)

            # normalize reward
            normalized_reward_default = normalize_rewards(task=task, absolute_reward=rewards_default)
            summed_up_reward_default += normalized_reward_default

            # set state in env
            # environment assumes a numpy array as state
            # copied_env_default.env_method("set_state", last_observation_default)

        if not done_optimal:
            # get action
            actions, _, _, _, _ = policy.get_action_and_value_and_forward_model_prediction(
                fm_network=fm_network,
                obs=torch.tensor(last_observation_state_optimal, device=device, dtype=torch.float32).clone().detach(),
                logger=logger,
                position_predicting=position_predicting,
                maximum_number_of_objects=maximum_number_of_objects)

            # we manually predict the next state
            # if position_predicting:
            #     last_observation_optimal = get_position_and_object_positions_of_observation(
            #         torch.tensor(last_observation_optimal, device=device),
            #         maximum_number_of_objects=maximum_number_of_objects,
            #         observation_width=observation_width, observation_height=observation_height, agent_size=agent_size)
            # else:
            #     last_observation_optimal = torch.tensor(last_observation_optimal, device=device,
            #                                             dtype=torch.float32).detach().clone()
            # forward_model_prediction_normal_distribution_optimal = fm_network(last_observation_optimal, actions.float())

            # get positions
            last_observation_optimal = get_position_and_object_positions_of_observation(
                torch.tensor(last_observation_state_optimal, device=device),
                maximum_number_of_objects=maximum_number_of_objects,
                observation_width=observation_width, observation_height=observation_height, agent_size=agent_size)
            # get next positions for new step with optimal action
            last_observation_optimal = get_next_position_observation_moonlander(
                observations=last_observation_optimal,
                actions=actions[0],
                observation_width=observation_width,
                observation_height=observation_height,
                agent_size=agent_size,
                maximum_number_of_objects=maximum_number_of_objects)

            # get reward from forward model prediction or environment
            # if reward_predicting and not use_reward_of_env:
            #     # state for env
            #     last_observation_optimal = np.expand_dims(
            #         get_observation_of_position_and_object_positions(agent_and_object_positions=
            #         forward_model_prediction_normal_distribution_optimal.mean[0][:-1].cpu().unsqueeze(
            #             0), observation_height=observation_height,
            #             observation_width=observation_width, agent_size=agent_size, task=task).flatten().cpu().numpy(),
            #         axis=0)
            #     rewards_optimal = np.expand_dims(forward_model_prediction_normal_distribution_optimal.mean[0][
            #                                          -1].cpu().detach().numpy(), axis=0)
            # else:
            x_position_of_agent = int(
                min(max(agent_size, last_observation_optimal[0][0]), observation_width - agent_size + 1))
            y_position_of_agent = int(last_observation_optimal[0][1])
            collected_objects = []
            for index in range(2, len(last_observation_optimal[0]), 2):
                if (
                        (
                                ((last_observation_optimal[0][index] - 1) == (x_position_of_agent - 1))
                                or ((last_observation_optimal[0][index] - 1) == x_position_of_agent)
                                or ((last_observation_optimal[0][index] - 1) == (x_position_of_agent + 1))
                                or (last_observation_optimal[0][index] == (x_position_of_agent - 1))
                                or (last_observation_optimal[0][index] == x_position_of_agent)
                                or (last_observation_optimal[0][index] == (x_position_of_agent + 1))
                                or ((last_observation_optimal[0][index] + 1) == (x_position_of_agent - 1))
                                or ((last_observation_optimal[0][index] + 1) == x_position_of_agent)
                                or ((last_observation_optimal[0][index] + 1) == (x_position_of_agent + 1))
                        )
                        and
                        (
                                ((last_observation_optimal[0][index + 1] - 1) == (y_position_of_agent - 1))
                                or ((last_observation_optimal[0][index + 1] - 1) == y_position_of_agent)
                                or ((last_observation_optimal[0][index + 1] - 1) == (y_position_of_agent + 1))
                                or (last_observation_optimal[0][index + 1] == (y_position_of_agent - 1))
                                or (last_observation_optimal[0][index + 1] == y_position_of_agent)
                                or (last_observation_optimal[0][index + 1] == (y_position_of_agent + 1))
                                or ((last_observation_optimal[0][index + 1] + 1) == (y_position_of_agent - 1))
                                or ((last_observation_optimal[0][index + 1] + 1) == y_position_of_agent)
                                or ((last_observation_optimal[0][index + 1] + 1) == (y_position_of_agent + 1))
                        )
                ):
                    collected_objects.append(
                        {'x': int(last_observation_optimal[0][index]), 'y': int(last_observation_optimal[0][index + 1]),
                         'size': agent_size})
            # state for env
            last_observation_state_optimal = np.expand_dims(
                get_observation_of_position_and_object_positions(agent_and_object_positions=last_observation_optimal,
                                                                 observation_height=observation_height,
                                                                 observation_width=observation_width,
                                                                 agent_size=agent_size,
                                                                 task=task).flatten().cpu().numpy(),
                axis=0)
            rewards_optimal, _ = calculate_gaussian_reward(
                state=last_observation_state_optimal.reshape(observation_height, observation_width + 2),
                collected_objects=collected_objects,
                agent_size=agent_size,
                task_type=task_type,
                current_reward_function="gaussian",
                x_position_of_agent=x_position_of_agent,
                y_position_of_agent=y_position_of_agent)

            # normalize reward
            normalized_reward_optimal = normalize_rewards(task=task, absolute_reward=rewards_optimal)
            summed_up_reward_optimal += normalized_reward_optimal

            # set state in env
            # environment assumes a numpy array as state
            copied_env_optimal.env_method("set_state", last_observation_optimal)

    # get a mean reward between 0 and 1
    summed_up_reward_default_normalized = summed_up_reward_default / (max(round(trajectory_length), 1))
    summed_up_reward_optimal_normalized = summed_up_reward_optimal / (max(round(trajectory_length), 1))

    # distance between the two trajectories
    difficulty = (max(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized)) - (
        min(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized))

    # difficulty is high if the rewards are quite different
    return difficulty, summed_up_reward_default_normalized


def normalize_rewards(task: str, absolute_reward) -> float:
    # normalize reward with MinMaxScaler
    if task == "dodge":
        # normalize reward with MinMaxScaler: (reward - min) / (max - min)
        # the maximum reward is 10 -> when no obstacles are in the area
        # the minimum reward is ~-100 -> when the agent is completely surrounded by obstacles
        # the minimum condition never happens, in general are most rewards between 0 and 10
        # to not have a normalization, where 99% of the numbers are similar
        # we choose to clip the smallest 1% -> which is a clipping from -100 to -3 -> clip to -3
        if absolute_reward < -3:
            absolute_reward = np.array([-3])
        # normalized_reward = (absolute_reward - (-3)) / (10 - (-3))
        # FIXME: try different normalization!!!
        # reward between 0 and 0.5
        normalized_reward = (((0.5 - 0) * (absolute_reward - (-3))) / (10 - (-3))) + 0
    elif task == "collect":
        # normalize reward with MinMaxScaler: (reward - min) / (max - min)
        # the maximum reward is ~350 -> when the agent is completely surrounded by coins
        # the minimum reward is ~0 -> when no coins are in the area
        # the maximum condition never happens, in general are most rewards between 0 and 60
        # to not have a normalization, where 99% of the numbers are similar
        # we choose to clip the highest 1% -> which is a clipping from 0 to 62 -> clip to 62
        if absolute_reward > 62:
            absolute_reward = np.array([62])
        # it is possible that the reward is slightly negative (e.g. -0.0000001), which breaks our normalization
        elif absolute_reward < 0:
            absolute_reward = np.array([0])
        # normalized_reward = (absolute_reward - 0) / (62 - 0)
        # FIXME: try different normalization!!!
        # reward between 0 and 0.5
        normalized_reward = (((1 - 0.5) * (absolute_reward - 0)) / (62 - 0)) + 0.5
    else:
        raise ValueError("Task {} not implemented".format(task))

    return normalized_reward
