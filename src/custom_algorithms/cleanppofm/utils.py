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


def get_summed_up_reward_of_env_with_predicted_states_hardcoded(env, last_observation_positions: torch.Tensor,
                                                                number_of_future_steps: int = 10) -> float:
    """
    Get the reward of the forward model prediction or environment through predicted states of the forward model
     for number_of_future_steps steps.
    Args:
        env: environment
        last_observation_positions: last known observation in form of positions
        number_of_future_steps: number of future steps to predict

    Returns:
        summed up reward of the forward model or environment for number_of_future_steps steps

    """
    env_name = env.env_method("get_wrapper_attr", "name")[0]
    task = env.env_method("get_wrapper_attr", "task")[0]
    observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
    observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
    agent_size = env.env_method("get_wrapper_attr", "size")[0]

    # default action is stay at same position
    if env_name == "MoonlanderWorldEnv":
        default_action = torch.tensor([[1]]).to(device)
    # using reward model of other envs is not implemented by now
    else:
        raise NotImplementedError(
            f"The current environment has not implemented the manual reward calculation for {number_of_future_steps} steps.")

    if task == "dodge":
        task_type = "obstacle"
    elif task == "collect":
        task_type = "coin"
    else:
        raise ValueError(f"The current task {task} is not supported.")

    summed_up_reward = 0
    for i in range(number_of_future_steps):
        # get observation state of positions
        last_observation_state = np.expand_dims(
            get_observation_of_position_and_object_positions(agent_and_object_positions=last_observation_positions,
                                                             observation_height=observation_height,
                                                             observation_width=observation_width,
                                                             agent_size=agent_size,
                                                             task=task).flatten().cpu().numpy(),
            axis=0)

        x_position_of_agent = int(
            min(max(agent_size, last_observation_positions[0][0]), observation_width - agent_size + 1))
        y_position_of_agent = int(last_observation_positions[0][1])
        collected_objects = get_collected_objects(observation_positions=last_observation_positions,
                                                  agent_size=agent_size, observation_width=observation_width)

        # calculate reward
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

        # problem: when collecting an object, the coin is removed after one step in the environment
        # here the coin stays the whole "collecting process" because we don't use the environment
        # therefore we get the collected objects and remove them, when they were collected
        if task == "collect":
            for collected_object in collected_objects:
                for index in range(2, len(last_observation_positions[0]), 2):
                    if collected_object['x'] == last_observation_positions[0][index] and collected_object['y'] == \
                            last_observation_positions[0][index + 1]:
                        last_observation_positions[0][index] = 0
                        last_observation_positions[0][index + 1] = 0

        # get next positions
        last_observation_positions = get_next_position_observation_moonlander(
            observations=last_observation_positions,
            actions=default_action[0],
            observation_width=observation_width,
            agent_size=agent_size)

    # normalize with mean of summed_up_reward
    return summed_up_reward / number_of_future_steps


def get_position_and_object_positions_of_observation(obs: torch.Tensor,
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
        observation_height: height of the observation
        agent_size: size of the agent in the observation
        # FIXME: note, these are the first objects you get, when going down in the obs and going from left to right
        # FIXME: these are not necessary the nearest objects to the agent

    Returns:
        position of the agent and maximum_number_of_objects objects in the observation, where the first two elements
        are the x and y position of the agent
    """
    agent_and_object_positions = []
    if agent_size > 2:
        raise ValueError(f"Get the positions of an observation is only supported for agent size <= 2, "
                         f"but you defined an agent size of {agent_size}.")
    if not obs.shape[1] == (observation_width + 2) * observation_height:
        raise ValueError(
            f"The given observation width {observation_width} and height {observation_height} "
            f"do not match the observation shape: {obs.shape}."
            f"The second observation shape element {obs.shape[1]} should be "
            f"(observation_width + 2) * observation_height = {(observation_width + 2) * observation_height}.")
    for obs_element in obs:
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
            # agent size is 2
            else:
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

            for index in indices_with_two_or_three:

                # get x and y coordinate of object
                # +2 because of the walls
                x_coordinate = (index % (observation_width + 2)) + agent_size - 1
                # get to the middle of the object
                y_coordinate = math.floor(index / (observation_width + 2))

                if agent_size == 2:
                    # check if object is only in the first line (then y_coordinate is -1) or also in the second line
                    if y_coordinate == 0:
                        # check if second line has also an object at the x position
                        if not (torch.all(
                                (obs_element[
                                 (x_coordinate + observation_width + 1):(x_coordinate + observation_width + 4)]
                                 == search_value) | (obs_element[
                                                     (x_coordinate + observation_width + 1):(
                                                             x_coordinate + observation_width + 4)] == 1))):
                            y_coordinate = -1
                    elif y_coordinate == (observation_height - 1):
                        if not (torch.all(
                                (obs_element[((x_coordinate - 1) + (observation_width + 2) * (observation_height - 2))
                                :((x_coordinate + 2) + (observation_width + 2) * (observation_height - 2))]
                                 == search_value) |
                                (obs_element[((x_coordinate - 1) + (observation_width + 2) * (observation_height - 2))
                                :((x_coordinate + 2) + (observation_width + 2) * (observation_height - 2))] == 1))):
                            y_coordinate = observation_height

                # remove agent from indices with two or three --> agent is added later at the beginning of the list
                if not (x_coordinate == (first_index_with_one + agent_size - 1) and y_coordinate == (agent_size - 1)):
                    x_y_coordinates.append([x_coordinate, y_coordinate])

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
                        )):
                    # object is left from agent
                    if current_x_coordinate < (first_index_with_one + agent_size - 1):
                        # check if object already started earlier
                        if obs_element.cpu()[current_x_coordinate - 2] == search_value:
                            x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])
                    elif current_x_coordinate > (first_index_with_one + agent_size - 1):
                        if obs_element.cpu()[current_x_coordinate + 2] == search_value:
                            x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])

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

            x_y_coordinates = x_y_coordinates_copy
            # add zeros to the list if we have not enough objects
            if len(x_y_coordinates) < maximum_number_of_objects:
                x_y_coordinates = x_y_coordinates + [[0, 0]] * (maximum_number_of_objects - len(x_y_coordinates))
            # else: remove objects if we have too many objects
            elif len(x_y_coordinates) > maximum_number_of_objects:
                x_y_coordinates = x_y_coordinates[:maximum_number_of_objects]

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


# outdated
def get_next_whole_observation(observations: torch.Tensor, actions: torch.Tensor, observation_width: int,
                               observation_height: int) -> torch.Tensor:
    """
    Calculate the next observation in the moonlander environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions
        observation_width: width of the observation
        observation_height: height of the observation

    Returns:
        next observation in the moonlander environment without input noise

    """
    raise NotImplementedError("This function is outdated.")
    if not observations.shape[1] == (observation_width + 2) * observation_height:
        raise ValueError(
            f"The given observation width {observation_width} and height {observation_height} "
            f"do not match the observation shape: {observations.shape}."
            f"The second observation shape element {observations.shape[1]} should be "
            f"(observation_width + 2) * observation_height = {(observation_width + 2) * observation_height}.")

    # object in observation is marked with 2 or 3
    if 2 in observations:
        search_value = 2
    else:
        search_value = 3

    # deep copy of next_observation
    observations_copy = observations.detach().clone()
    # remove agent and objects
    observations_copy[observations_copy == 1] = 0
    observations_copy[observations_copy == 2] = 0

    # get x coordinates of agent for each batch element
    x_index_of_agent = torch.nonzero(observations == 1, as_tuple=True)[1]
    # get x, y coordinates of objects for each batch element (index of batch element in element_in_batch)
    element_in_batch, indices_of_objects = torch.nonzero(observations == search_value, as_tuple=True)
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
    observations_copy[valid_element_in_batch, valid_new_indices_of_objects] = 2

    # add agent to next observation
    new_x_index_of_agent = torch.clamp(x_index_of_agent + (actions - 1), min=1, max=observation_width)
    observations_copy[torch.arange(observations.shape[0]), new_x_index_of_agent] = 1

    return observations_copy


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
            y_position_of_object = int(torch.round(copy_of_agent_and_object_position[counter + 1]))
            if (agent_size <= x_position_of_object <= observation_width + 1 - agent_size) and (
                    -agent_size + 1 <= y_position_of_object <= observation_height):
                matrix[
                # objects can also fly into the grid
                max(
                    0,
                    min(
                        observation_height - 1,
                        int(torch.round(copy_of_agent_and_object_position[counter + 1]) - agent_size + 1)
                    )
                ):  # y start of object
                # objects can also fly out of the grid
                max(
                    0,
                    min(
                        observation_height - 1,
                        int(torch.round(copy_of_agent_and_object_position[counter + 1]) + agent_size - 1)
                    )
                ) + 1,  # y end of object
                x_position_of_object - agent_size + 1:  # x start of object
                x_position_of_object + agent_size  # x end of object
                ] = object_value
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

    # form observations list to numpy array for faster calculations
    observations = np.array(observations)

    observations_tensor = torch.flatten(torch.tensor(observations, device=device, dtype=torch.float32), start_dim=1)

    return observations_tensor


def get_next_position_observation_moonlander(observations: torch.Tensor, actions: torch.Tensor, observation_width: int,
                                             agent_size: int) -> torch.Tensor:
    """
    Calculate the next observation in the moonlander environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions
        observation_width: width of the observation
        agent_size: size of the agent in the observation

    Returns:
        next observation in the moonlander environment without input noise
    """
    next_observation_without_input_noise = observations.clone().detach()

    # loop through every observation in batch
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

        # apply step to every object y position
        next_observation_without_input_noise[index][3::2] -= 1

        # check if there is an object that already is now on position -2 (for agent size 2) or -3 (for agent size 3)
        # after doing a step -> remove it
        while counter <= (observations.shape[1] - 2):
            if not next_observation_without_input_noise[index][counter] == 0 and \
                    next_observation_without_input_noise[index][counter + 1] == -agent_size:
                next_observation_without_input_noise[index][counter] = 0
                next_observation_without_input_noise[index][counter + 1] = 0
            # clamp y-position of empty object positions back to 0
            elif next_observation_without_input_noise[index][counter] == 0 and \
                    next_observation_without_input_noise[index][counter + 1] == -1:
                next_observation_without_input_noise[index][counter + 1] = 0
            counter += 2

    return next_observation_without_input_noise


def calculate_prediction_error(env_name, next_obs_positions, forward_model_prediction_normal_distribution: torch.normal,
                               first_possible_x_position: int, last_possible_x_position: int) -> float:
    """
    Calculate the prediction error between the next obs and the forward model prediction.
    Args:
        env_name: name of the environment
        next_obs_positions: observation in positions after actually executing the action
        forward_model_prediction_normal_distribution: prediction of next observation by forward model
        first_possible_x_position: first possible x position of the agent
        last_possible_x_position: last possible x position of the agent

    Returns:
        prediction error between the next obs and the forward model prediction
    """
    ##### CALCULATE PREDICTION ERROR #####
    # prediction error version one -> standard deviation
    # prediction_error = forward_normal.stddev.mean().item()
    # prediction error version two -> Euclidean distance
    # calculate manually prediction error (Euclidean distance)
    if env_name == "MoonlanderWorldEnv":
        # we just care for the x position of the moonlander agent, because the y position is always equally
        # to the size of the agent
        # independently of using the whole obs or the position prediction,
        # we use the position predictions to calculate the prediction error

        # maximal distance possible in moonlander world to use min/max scaler
        max_distance_in_moonlander_world = math.sqrt(
            (last_possible_x_position - first_possible_x_position) ** 2)

        predicted_x_position = torch.tensor([min(max(first_possible_x_position,
                                                     forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[
                                                         0][0]),
                                                 last_possible_x_position)], device=device)
        prediction_error = (math.sqrt(
            torch.sum((predicted_x_position - next_obs_positions[0][0]) ** 2))) / max_distance_in_moonlander_world
    else:
        raise ValueError("Environment not supported")

    return prediction_error


def calculate_need_for_control(env, policy, fm_network, logger, position_predicting: bool, prediction_error: float,
                               maximum_number_of_objects: int = 5, last_observation_state: np.array = None) -> tuple[
    float, float]:
    """
    Calculate the need for control of the environment by simulating the default trajectory
    and the "optimal" trajectory the agent would choose.
    Args:
        env: environment
        policy: agent policy to predict actions
        fm_network: forward model network
        logger: logger
        prediction_error: error between predicted and last actual observation
        position_predicting: if the forward model is predicting the position or actual observation
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction
        last_observation_state: last observation state can be given and not selected by the environment
    Returns:
        need for control between 0 and 1
        summed up rewards when executing the default action (trajectory length is calculated by prediction error)
    """
    env_name = env.env_method("get_wrapper_attr", "name")[0]
    # default action is stay at same position
    if env_name == "MoonlanderWorldEnv":
        default_action = torch.tensor([[1]]).to(device)
    # using reward model of other envs is not implemented by now
    else:
        raise ValueError(
            "The current environment does not support the need for control calculation.")

    if not position_predicting:
        raise NotImplementedError("Using the actual states instead of positions is not implemented yet.")

    observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
    observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
    agent_size = env.env_method("get_wrapper_attr", "size")[0]
    task = env.env_method("get_wrapper_attr", "task")[0]

    if task == "dodge":
        task_type = "obstacle"
    elif task == "collect":
        task_type = "coin"
    else:
        raise ValueError(f"The current task {task} is not supported.")

    # calculate the trajectory lengths through the prediction error
    # we decide that the trajectory length is half the observation size of the environment
    # when the prediction error is 0
    trajectory_length = calculate_trajectory_length(observation_height=observation_height,
                                                    prediction_error=prediction_error)

    if last_observation_state is None:
        last_observation_state_default = np.expand_dims(env.env_method("get_wrapper_attr", "state")[0].flatten(),
                                                        axis=0)
        last_observation_state_optimal = copy.deepcopy(last_observation_state_default)
    else:
        last_observation_state_default = last_observation_state
        last_observation_state_optimal = copy.deepcopy(last_observation_state_default)

    ##### CALCULATE REWARDS THROUGH ENVIRONMENT #####
    summed_up_reward_default = 0
    summed_up_reward_optimal = 0

    # simulate at least one step
    for i in range(max(round(trajectory_length), 1)):
        ##### DEFAULT ACTION #####
        normalized_reward_default, last_observation_state_default = get_next_normalized_reward(
            last_observation_state=last_observation_state_default,
            action=default_action[0],
            maximum_number_of_objects=maximum_number_of_objects,
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size, task=task, task_type=task_type)
        summed_up_reward_default += normalized_reward_default

        ##### OPTIMAL ACTION #####
        # get optimal action of agent
        actions, _, _, _, _ = policy.get_action_and_value_and_forward_model_prediction(
            fm_network=fm_network,
            obs=torch.tensor(last_observation_state_optimal, device=device, dtype=torch.float32).clone().detach(),
            logger=logger,
            position_predicting=position_predicting,
            maximum_number_of_objects=maximum_number_of_objects)

        normalized_reward_optimal, last_observation_state_optimal = get_next_normalized_reward(
            last_observation_state=last_observation_state_optimal,
            action=actions[0],
            maximum_number_of_objects=maximum_number_of_objects,
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size, task=task, task_type=task_type)
        summed_up_reward_optimal += normalized_reward_optimal

    # get a mean reward between 0 and 1
    summed_up_reward_default_normalized = summed_up_reward_default / (max(round(trajectory_length), 1))
    summed_up_reward_optimal_normalized = summed_up_reward_optimal / (max(round(trajectory_length), 1))

    # distance between the two trajectories
    need_for_control = (max(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized)) - (
        min(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized))

    # need for control is high if the rewards are quite different
    return need_for_control, summed_up_reward_default_normalized


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
        elif absolute_reward > 10:
            raise ValueError("Reward should not be higher than 10.")
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
        raise NotImplementedError("Task {} not implemented".format(task))

    return normalized_reward


def get_collected_objects(observation_positions: torch.tensor, agent_size: int, observation_width: int) -> list[
    dict[str, int]]:
    if agent_size > 2:
        raise NotImplementedError("Agent size > 2 is not supported to get the collected objects in the positions.")
    x_position_of_agent = int(
        min(max(agent_size, observation_positions[0][0]), observation_width - agent_size + 1))
    y_position_of_agent = int(observation_positions[0][1])
    collected_objects = []

    for index in range(2, len(observation_positions[0]), 2):
        if not (observation_positions[0][index] == 0 and observation_positions[0][index + 1] == 0):

            if agent_size == 1:
                if (observation_positions[0][index] == x_position_of_agent) and (
                        observation_positions[0][index + 1] == y_position_of_agent):
                    collected_objects.append(
                        {'x': int(observation_positions[0][index]),
                         'y': int(observation_positions[0][index + 1]),
                         'size': agent_size})
            # agent size of 2
            else:
                if (
                        (
                                ((observation_positions[0][index] - 1) == (x_position_of_agent - 1))
                                or ((observation_positions[0][index] - 1) == x_position_of_agent)
                                or ((observation_positions[0][index] - 1) == (x_position_of_agent + 1))
                                or (observation_positions[0][index] == (x_position_of_agent - 1))
                                or (observation_positions[0][index] == x_position_of_agent)
                                or (observation_positions[0][index] == (x_position_of_agent + 1))
                                or ((observation_positions[0][index] + 1) == (x_position_of_agent - 1))
                                or ((observation_positions[0][index] + 1) == x_position_of_agent)
                                or ((observation_positions[0][index] + 1) == (x_position_of_agent + 1))
                        )
                        and
                        (
                                ((observation_positions[0][index + 1] - 1) == (y_position_of_agent - 1))
                                or ((observation_positions[0][index + 1] - 1) == y_position_of_agent)
                                or ((observation_positions[0][index + 1] - 1) == (y_position_of_agent + 1))
                                or (observation_positions[0][index + 1] == (y_position_of_agent - 1))
                                or (observation_positions[0][index + 1] == y_position_of_agent)
                                or (observation_positions[0][index + 1] == (y_position_of_agent + 1))
                                or ((observation_positions[0][index + 1] + 1) == (y_position_of_agent - 1))
                                or ((observation_positions[0][index + 1] + 1) == y_position_of_agent)
                                or ((observation_positions[0][index + 1] + 1) == (y_position_of_agent + 1))
                        )
                ):
                    collected_objects.append(
                        {'x': int(observation_positions[0][index]),
                         'y': int(observation_positions[0][index + 1]),
                         'size': agent_size})

    return collected_objects


def calculate_trajectory_length(observation_height: int, prediction_error: float) -> float:
    # calculate the trajectory lengths through the prediction error
    # we decide that the trajectory length is half the observation size of the environment
    # when the prediction error is 0
    return - (observation_height / 2) * prediction_error + observation_height / 2


def get_next_normalized_reward(last_observation_state: torch.Tensor, action: torch.Tensor,
                               maximum_number_of_objects: int, observation_width: int, observation_height: int,
                               agent_size: int, task: str, task_type: str) -> tuple[float, torch.Tensor]:
    if not last_observation_state.shape[1] == (observation_width + 2) * observation_height:
        raise ValueError(
            f"The given observation width {observation_width} and height {observation_height} "
            f"do not match the observation shape: {last_observation_state.shape}."
            f"The second observation shape element {last_observation_state.shape[1]} should be "
            f"(observation_width + 2) * observation_height = {(observation_width + 2) * observation_height}.")
    if not (task == "dodge" or task == "collect"):
        raise NotImplementedError(f"The task {task} is not supported.")
    if not (task_type == "obstacle" or task_type == "coin"):
        raise NotImplementedError(f"The task type {task_type} is not supported.")
    if not ((task == "dodge" and task_type == "obstacle") or (task == "collect" and task_type == "coin")):
        raise NotImplementedError(f"The task {task} and task type {task_type} combination is not supported.")

    # get positions of last observation
    last_observation = get_position_and_object_positions_of_observation(
        torch.tensor(last_observation_state, device=device), maximum_number_of_objects=maximum_number_of_objects,
        observation_width=observation_width, observation_height=observation_height, agent_size=agent_size)

    # get next positions for new step with action
    last_observation = get_next_position_observation_moonlander(
        observations=last_observation,
        actions=action,
        observation_width=observation_width,
        agent_size=agent_size)

    # get collected objects to calculate reward
    x_position_of_agent = int(
        min(max(agent_size, last_observation[0][0]), observation_width - agent_size + 1))
    y_position_of_agent = int(last_observation[0][1])
    collected_objects = get_collected_objects(observation_positions=last_observation, agent_size=2,
                                              observation_width=observation_width)

    # determine state of new positions for reward calculation
    last_observation_state = np.expand_dims(
        get_observation_of_position_and_object_positions(agent_and_object_positions=last_observation,
                                                         observation_height=observation_height,
                                                         observation_width=observation_width,
                                                         agent_size=agent_size,
                                                         task=task).flatten().cpu().numpy(),
        axis=0)

    # calculate reward
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

    return normalized_reward, last_observation_state
