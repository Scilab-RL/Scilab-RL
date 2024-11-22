import random as rnd
from typing import List, Dict, Tuple
import copy
import math
import cv2
import numpy as np
import scipy.stats


def create_ranges_of_objects_funnels_and_drifts(
        world_x_width: int = 80,
        world_y_height: int = 2100,
        height_padding_areas: int = 60,
        level_difficulty: str = "easy",
        drift_length: int = 5,
        agent_size: int = 1,
        use_variable_drift_intensity: bool = False,
        invisible_drift_probability: float = 0.0,
        fake_drift_probability: float = 0.0,
        funnel_range: bool = True,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    creates a list of ranges where objects occur
    + list of ranges where no objects occur (for funnels)
    + list of drifts
    There are three different level difficulties: easy, middle, hard, empty
    Args:
        world_x_width: width of the world (excluding walls)
        world_y_height: height of whole world
        height_padding_areas: height of the starting and ending area (individually)
        level_difficulty: easy, middle or hard
        drift_length: lengths of drift
        agent_size: size of agent
        use_variable_drift_intensity: Whether to use stronger (intensity 1) and weaker (intensity 3) drift types
        invisible_drift_probability: The probability with which drifts should be marked as invisible to the agent.
        fake_drift_probability: Probability of a given drift being fake (displayed but zero intensity). The sum with
                        invisible_drift_probability must be <= 1.
        funnel_range: Whether to create funnel ranges
    Returns:
        list of dicts of objects, list of free ranges, list of drifts
    """

    # in the first observation are no extras like obstacles, funnels or drifts, therefore this part is removed
    # we will add it in the end again
    y_height_without_padding_area = world_y_height - 2 * height_padding_areas

    # the world is divided in sections where obstacles are placed and where not
    # where the obstacles are not places, there is a funnel defined

    if level_difficulty == "empty":
        # no obstacles, no drift, no funnel
        return [], [], []
    elif level_difficulty == "easy":
        number_of_drifts = int(y_height_without_padding_area / 100)
    elif level_difficulty == "middle":
        number_of_drifts = int(y_height_without_padding_area / 100)
    elif level_difficulty == "hard":
        number_of_drifts = int(y_height_without_padding_area / 50)
    else:
        raise KeyError(
            "Leveldifficulty",
            level_difficulty,
            'is not defined. Please use a leveldifficulty out of ["no", "easy", "middle", "hard"]',
        )

    if funnel_range:
        # divide world in four parts --> 1. funnel, 1. obstacles phase , 2. funnel, 2. obstacles phase
        section_length = y_height_without_padding_area / 4

        # we define the minimum x width as two times the agent size + two additional spaces
        # the funnel can not be more tight than this in the middle of the funnel
        minimum_x_width = 2 * agent_size + 2

        # these are the maximum number of y values (range) the funnel can have without getting too tight
        # for example, we have a x width of size 9 and an agent size of 1, then the minimum_x_width is 4
        # this means the funnels range only has to be up to 5 because it cannot tight the funnel more
        # than in the 3 spaces building up and 2 building down
        # 1  2  3  4  5  6  7  8  9
        # -1 0  0  0  0  0  0  0 -1
        # -1 -1 0  0  0  0  0 -1 -1
        # -1 -1 0  0  0  0  0 -1 -1
        # -1 -1 -1 0  0  0 -1 -1 -1
        # -1 0  0  0  0  0  0  0 -1
        # it is not possible to make the funnel more tight than this
        maximum_needed_range_for_funnel = world_x_width - minimum_x_width

        # when the range of the funnel for getting to the tightest point is too big for fitting into the world
        # then just divide the world into 4 equal parts
        # for the example above, all y height without starting area that is below or equal to 20 will
        # divide the world into 4 equal parts, because each part is less or equal to 5 (which is the maximum needed range)
        # but if the world is bigger than 20, then the world will be divided into 4 parts but the range of the funnel parts
        # is only 5 because it doesn't need to be longer
        if maximum_needed_range_for_funnel >= section_length:
            ### OBJECTS
            object_range_list = [
                [
                    section_length + 1,
                    section_length * 2,
                ],
                [
                    section_length * 3 + 1,
                    y_height_without_padding_area,
                ],
            ]
            ### WALLS
            free_range_list = [
                [1, section_length],
                [
                    section_length * 2 + 1,
                    section_length * 3,
                ],
            ]
        # but when the funnel can get to the tightest point
        # then we want to make sure that the range of the funnel is only that long how it needs to be
        else:
            # If the funnel section is longer than the funnel needs, the obstacle range gets extended into
            # the empty space.
            # the remaining part of the world can be used for the obstacles phase
            # we need two times the funnel (2*maximum_needed_range_for_funnel)
            # the remaining is for obstacles (y_height_without_starting_area - 2*maximum_needed_range_for_funnel)
            # we divide this by 2 because we have two phases of obstacles
            remaining_range_for_objects = int(
                (y_height_without_padding_area - (2 * maximum_needed_range_for_funnel)) / 2
            )

            ### OBJECTS
            object_range_list = [
                [
                    maximum_needed_range_for_funnel + 1,
                    maximum_needed_range_for_funnel + remaining_range_for_objects,
                ],
                [
                    (2 * maximum_needed_range_for_funnel) + remaining_range_for_objects + 1,
                    y_height_without_padding_area,
                ],
            ]
            ### WALLS
            free_range_list = [
                [1, maximum_needed_range_for_funnel],
                [
                    maximum_needed_range_for_funnel + remaining_range_for_objects + 1,
                    (2 * maximum_needed_range_for_funnel) + remaining_range_for_objects,
                ],
            ]
    else:
        # If there are no funnels, objects can be placed everywhere
        object_range_list = [
            [agent_size * 3, y_height_without_padding_area],
        ]
        free_range_list = []

    ### DRIFT
    drift_range_list = create_drift_ranges(
        world_y_height=y_height_without_padding_area,
        number_of_drifts=number_of_drifts,
        drift_length=drift_length,
        safe_ranges=free_range_list,
        use_variable_drift_intensity=use_variable_drift_intensity,
        invisible_drift_probability=invisible_drift_probability,
        fake_drift_probability=fake_drift_probability,
    )

    # append starting area to all obstacle ranges
    for object_range in object_range_list:
        object_range[0] = int(object_range[0] + height_padding_areas)
        object_range[1] = int(object_range[1] + height_padding_areas)

    # append starting area to all free ranges
    for free_range in free_range_list:
        free_range[0] = int(free_range[0] + height_padding_areas)
        free_range[1] = int(free_range[1] + height_padding_areas)

    # append starting area to all drift ranges
    for drift_range in drift_range_list:
        drift_range[0] = int(drift_range[0] + height_padding_areas)
        drift_range[1] = int(drift_range[1] + height_padding_areas)

    return object_range_list, free_range_list, drift_range_list


### OBSTACLES


def create_list_of_object_dicts(
        object_range_list: List[List[int]] = None,
        object_size: int = 1,
        world_x_width: int = 80,
        level_difficulty: str = "no",
        normalized_object_placement: bool = False,
        allow_overlapping_objects: bool = True,
        number_of_objects: int = None,
) -> List[Dict[str, int]]:
    """
    creates a list of object dictionaries; the objects are in the ranges defined in the hand overed ranges list
    Args:
        object_range_list: list of two-element lists which indicate the ranges where the objects occur
        object_size: size of the objects
        world_x_width: width of the world (excluding walls)
        level_difficulty: easy, middle, hard or no
        normalized_object_placement: if True, the obstacle is placed more in the middle
        allow_overlapping_objects: Whether to allow overlapping obstacles (only applies if size > 1).
                                        Setting this to false slows training down significantly.

    Returns: list of dictionaries which define an obstacle

    """
    if level_difficulty not in ["empty", "easy", "middle", "hard"]:
        raise KeyError(
            "Leveldifficulty",
            level_difficulty,
            'is not defined. Please use a leveldifficulty out of ["empty", "easy", "middle", "hard"]',
        )
    if level_difficulty == "empty":
        return []

    object_dict_list = []
    if number_of_objects:
        if number_of_objects < 0:
            raise ValueError("The number of objects must be >= 0.")
        if number_of_objects > 0 and object_range_list is None:
            raise ValueError(
                "If the number of objects is > 0, object_range_list must be specified."
            )
        number_of_objects = int(number_of_objects / len(object_range_list))

    # loop through the list of all ranges where obstacles occur
    for object_range in object_range_list:
        if not number_of_objects:
            length_of_object_range = len(range(object_range[0], object_range[1]))
            if level_difficulty == "easy":
                number_of_objects = int(length_of_object_range / 25)
            elif level_difficulty == "middle":
                number_of_objects = int(length_of_object_range / 10)
            elif level_difficulty == "hard":
                number_of_objects = int(length_of_object_range / 5)

        # the number of obstacles indicates how many obstacles are in the range
        # check that there is enough space for the agent to pass obstacles
        if number_of_objects == 0:
            return []

        object_locations = generate_object_field(
            number_of_objects=number_of_objects,
            object_size=object_size,
            x_width=world_x_width,
            range_start=object_range[0],
            range_end=object_range[1],
            normalized_object_placement=normalized_object_placement,
            allow_overlapping_objects=allow_overlapping_objects,
        )

        for location_vector in object_locations:
            object_dict_list.append(
                {
                    # Add 1 for the walls
                    "x": location_vector[0] + 1,
                    "y": location_vector[1],
                    "size": object_size,
                }
            )

    # sort obstacles list by y value of obstacles
    object_dict_list = sorted(object_dict_list, key=lambda dictionary: dictionary["y"])
    return object_dict_list


def generate_object_field(
        number_of_objects: int,
        object_size: int = 1,
        x_width: int = 80,
        range_start: int = 1,
        range_end: int = 5,
        normalized_object_placement: bool = False,
        allow_overlapping_objects: bool = True,
) -> np.array:
    """
    Generates an array of 2D vectors, each indicating the location for an obstacle.
    Args:
        number_of_objects: Number of objects to be generated.
        object_size: Size of the objects to be generated (used to avoid collisions).
        x_width: Width of the world.
        range_start: Start of the range to generate objects in.
        range_end: End of the range to generate objects in.
        normalized_object_placement: Whether to distribute objects according to a
                                        normal distribution along the x axis.
        allow_overlapping_objects: Whether to allow overlapping objects (only applies if size > 1).
                                        Setting this to false slows training down significantly.

    Returns: An aray with object locations.

    """

    # General idea of this algorithm: Assign a probability to each cell, and sample from that probability table
    # without replacement to get obstacle locations.
    # If we also avoid overlapping obstacles, we set the probabilities for all cells to zero where generating an
    # obstacle would overlap with an existing one. This isn't vectorized and thus much slower.

    range_length = range_end - range_start

    cell_count = range_length * x_width

    # Space an object needs around itself
    object_padding = object_size - 1

    # Remove columns where the obstacle would clip into a wall
    cell_count -= 2 * object_padding * range_length

    # Remove upper/lower rows where obstacles might clip into disallowed territory
    # We subtract 2 from the width because those cells were already deleted in the previous step
    cell_count -= 2 * object_padding * (x_width - 2)

    # Width/Height of the area where obstacles may be generated (this excludes safety margins at the walls)
    generated_width = x_width - 2 * object_padding
    generated_height = range_length - 2 * object_padding

    if normalized_object_placement:
        row_probabilities = scipy.stats.binom.pmf(
            k=np.arange(generated_width), p=0.5, n=generated_width
        )

        # Divide probabilities by number of rows so the total probability over all rows sums to 1
        # We also multiply by row_probabilities.sum() as the sum for one row doesn't quite equal 1 otherwise
        row_probabilities = np.divide(
            row_probabilities, generated_height * row_probabilities.sum()
        )

        # Repeat the probabilities of one row, for each row
        probability_table = np.resize(
            row_probabilities, new_shape=generated_width * generated_height
        )

    else:
        if allow_overlapping_objects:
            probability_table = None
        else:
            # Unnormalized uniform probabilities, because they'll be normalized later on.
            probability_table = np.ones(shape=(generated_width, generated_height))

    # Generate obstacles using numpy vectorization
    if allow_overlapping_objects:
        object_indexes = np.random.choice(
            a=cell_count, size=number_of_objects, p=probability_table, replace=False
        )

    # Generate objects one by one
    else:
        object_indexes = np.zeros(number_of_objects, dtype=np.int32)

        for i in range(number_of_objects):
            total_weight = probability_table.sum()
            if total_weight == 0:
                print("Warning: Insufficient space to place all obstacles!")
                object_indexes.resize((i - 1,))
                break

            # Normalize weights to get probabilities
            probability_table = np.divide(probability_table, total_weight)

            object_indexes[i] = np.random.choice(a=cell_count, p=probability_table.flat)

            # Set probabilities to zero for all cells where another obstacle
            # would collide with this one
            x, y = np.unravel_index(
                indices=object_indexes[i], shape=(generated_width, generated_height)
            )

            # Range of x values that must not get another obstacle to avoid collisions (max exclusive)
            # min/max to avoid going over the array bounds
            min_invalid_x = max(x + 2 * (-object_size + 1), 0)
            max_invalid_x = min(x + 2 * object_size - 1, generated_width)

            # Range of y values that must not get another obstacle to avoid collisions (max exclusive)
            # min/max to avoid going over the array bounds
            min_invalid_y = max(y + 2 * (-object_size + 1), 0)
            max_invalid_y = min(y + 2 * object_size - 1, generated_height)

            # Set the corresponding ranges to 0
            probability_table[
            min_invalid_x:max_invalid_x, min_invalid_y:max_invalid_y
            ] = 0

    # Convert flattened indices back to 2D indices
    x, y = np.unravel_index(
        indices=object_indexes,
        shape=(x_width - 2 * object_padding, range_length - 2 * object_padding),
    )
    y = np.add(y, range_start + object_padding)
    x = np.add(x, object_padding)

    # Convert vectors of x and y indices into an array of 2D-vectors
    return np.stack(arrays=(x, y), axis=1)


### WALLS
def create_dict_of_world_walls(
        list_of_free_ranges: List[List[int]],
        world_y_height: int = 2100,
        world_x_width: int = 80,
        agent_size: int = 1,
        no_crashes: bool = False,
) -> Dict[str, List[int]]:
    """
    creates a dict with every y in the world as a key, the value of each key describes where the wall is,
    when it is [0, observation_space_x], there is no funnel, smaller values indicate a funnel
    Args:
        list_of_free_ranges: list of two-element lists with starting and ending free range
        world_y_height: height of world
        world_x_width: width of the world (excluding walls)
        agent_size: size of the agent
        no_crashes: defines whether the agent should be able to crash into a wall and obstacles (if yes: create funnels, otherwise not)

    Returns: dict with every y value and its corresponding wall value

    """
    wall_dict = dict()
    if not no_crashes:
        for free_range in list_of_free_ranges:
            # the funnel gets smaller and wider in the complete free range
            # the lengths of the funnel indicates one of this cones
            lengths_of_funnel = int((len(range(free_range[0], free_range[1]))) / 2)
            current_wall_change_number = 0
            counter = 0
            count_same_size = 0

            for index in range(free_range[0], free_range[1]):

                # x and y value of the wall
                wall_dict[str(index)] = [
                    0 + current_wall_change_number,
                    world_x_width + 1 - current_wall_change_number,
                ]

                # check if the funnel should get smaller or wider
                if counter < lengths_of_funnel:
                    # this if prevents from making the funnel too small --> the agent still has to pass
                    if len(
                            range(
                                current_wall_change_number,
                                world_x_width + 1 - current_wall_change_number,
                            )
                    ) >= (agent_size + 2):
                        current_wall_change_number += 1
                    else:
                        count_same_size += 1
                else:
                    # this if prevents from making the funnel two small --> the agent still has to pass
                    if count_same_size > 0:
                        count_same_size -= 1
                    else:
                        current_wall_change_number -= 1

                counter += 1

    # create dict entry for all y values that are not covered in the free ranges
    for index in range(1, world_y_height + 1):
        if str(index) not in wall_dict:
            wall_dict[str(index)] = [0, world_x_width + 1]

    # sorting
    wall_dict_int_keys = {int(k): v for k, v in wall_dict.items()}
    sorted_wall_dict = dict()
    for wall_tuple in sorted(wall_dict_int_keys.items()):
        sorted_wall_dict[str(wall_tuple[0])] = wall_tuple[1]

    return sorted_wall_dict


### DRIFT
def create_drift_ranges(
        world_y_height: int = 2100,
        number_of_drifts: int = 0,
        drift_length: int = 5,
        safe_ranges: List[List[int]] = [],
        use_variable_drift_intensity: bool = False,
        invisible_drift_probability: float = 0.0,
        fake_drift_probability: float = 0.0,
) -> List[List[int]]:
    """
    creates ranges where drift occurs
    Args:
        world_y_height: height of the world
        number_of_drifts: number of drifts that occur in the world
        drift_length: length of the drift
        safe_ranges: ranges where no drift should be generated
        use_variable_drift_intensity: Whether to use variable drift strength. If this is off, drift of
                        intensity 1 is always generated. Otherwise, drift of intensity 1, 2 and 3 is generated.
                        Drift of intensity 2 happens at every other step, intensity 3 only at every third step.
        invisible_drift_probability: Probability of a given drift being invisible. The sum with fake_drift_probability
                        must be <= 1.
        fake_drift_probability: Probability of a given drift being fake (displayed but zero intensity). The sum with
                        invisible_drift_probability must be <= 1.

    Returns: a list of five-element lists describing start, stop, direction/intensity and visibility of each drift range
             and if it is fake

    """
    drift_ranges = []
    length_of_safe_ranges = 0
    for safe_range in safe_ranges:
        length_of_safe_ranges += safe_range[1] - safe_range[0]

    # ensure that the number of drifts with its lengths fit in the world
    if (number_of_drifts * drift_length) > (world_y_height - length_of_safe_ranges):
        raise EnvironmentError(
            f"The world height {world_y_height} is too small for implementing {number_of_drifts} drifts".format(
                world_y_height=world_y_height, number_of_drifts=number_of_drifts
            )
        )

    for drift_number in range(number_of_drifts):
        counter = 0
        not_found_range = True
        while not_found_range:
            if counter == 100:
                raise EnvironmentError(
                    f"Could not find a valid drift range after {counter} tries.".format(
                        counter=counter
                    )
                )
            # define a random drift range
            random_drift_starting_position = rnd.randrange(
                start=1, stop=world_y_height - drift_length + 1
            )
            random_drift_range = range(
                random_drift_starting_position,
                random_drift_starting_position + drift_length,
            )

            # check if new drift is overlapping with already existing drifts or safe ranges
            overlapping_with_disallowed_range = False

            disallowed_ranges = drift_ranges + safe_ranges

            for disallowed_range in disallowed_ranges:
                if not (
                        len(
                            range(
                                max(random_drift_range[0], disallowed_range[0]),
                                # random_drift_range is a range object, so the last element is needed
                                # disallowed_range is a list, so we need the second element and not the last one
                                # because there are other elements in the list
                                min(random_drift_range[-1] + 1, disallowed_range[1]),
                            )
                        )
                        == 0
                ):
                    overlapping_with_disallowed_range = True
                    counter += 1
                    break

            if not overlapping_with_disallowed_range:
                random_number = rnd.random()
                is_visible = random_number >= invisible_drift_probability
                is_fake = (
                        0
                        < random_number - invisible_drift_probability
                        <= fake_drift_probability
                )

                # the drifts randomly choose between a drift to the right and a drift to the left
                if not use_variable_drift_intensity:
                    drift_direction_and_intensity = rnd.choice([-1, 1])
                else:
                    drift_direction_and_intensity = rnd.choice([-3, -2, -1, 1, 2, 3])

                drift_ranges.append(
                    [
                        random_drift_starting_position,
                        random_drift_starting_position + drift_length,
                        drift_direction_and_intensity,
                        is_visible,
                        is_fake,
                    ]
                )
                not_found_range = False

    drift_ranges = sorted(drift_ranges, key=lambda x: x[0])
    return drift_ranges


def create_agent_observation(
        following_observation_size: int,
        drift_ranges: List[List[int]],
        walls_dict: Dict[str, List[int]],
        object_dict_list: List[Dict[str, int]],
        agent_x_position: int,
        agent_y_position: int,
        world_x_width: int,
        agent_size: int = 1,
        object_type: str = "obstacle",
        no_crashes: bool = False,
) -> np.ndarray:
    """
    creates the agent observation (matrix)
    Args:
        following_observation_size: number of lines of the agent observation matrix
        drift_ranges: dictionary with every y and its corresponding drift
        walls_dict: dictionary with every y value and its corresponding wall value
        object_dict_list: list of dictionaries of obstacles (x, y, size)
        agent_x_position: current x position of the agent
        agent_y_position: current y position of the agent
        world_x_width: width of the world (excluding walls)
        agent_size: size of the agent
        object_type: Which type the object passed have ('obstacle' or 'coin'). Obstacles are rendered as a -1, coins as
                    a 2.
        no_crashes: defines whether the agent should be able to crash into a wall and obstacles

    Returns: returns an agent observation matrix of shape: (channels, height, width)

    """

    current_relevant_object_dict_list = find_visible_objects(
        following_observation_size=following_observation_size,
        object_dict_list=object_dict_list,
        agent_y_position=agent_y_position,
        agent_size=agent_size,
    )

    # The width + 2 is because we have entries for the walls on the left and right, which don't count
    # towards the world width.
    matrix = np.zeros(
        shape=(following_observation_size, world_x_width + 2), dtype=np.int16
    )

    add_drift_to_observation(
        observation_start_row=agent_y_position - agent_size + 1,
        following_observation_size=following_observation_size,
        drift_ranges=drift_ranges,
        matrix=matrix,
    )

    for index in range(following_observation_size):
        matrix_row = matrix[index, :]

        current_y_position = agent_y_position + index - agent_size + 1

        add_funnels_to_observation(
            current_y_position=current_y_position,
            walls_dict=walls_dict,
            matrix_row=matrix_row,
        )

        if object_type == "obstacle":
            add_objects_to_observation(
                current_relevant_object_dict_list=current_relevant_object_dict_list,
                current_y_position=current_y_position,
                matrix_row=matrix_row,
                symbol=3,
            )
        else:
            add_objects_to_observation(
                current_relevant_object_dict_list=current_relevant_object_dict_list,
                current_y_position=current_y_position,
                matrix_row=matrix_row,
                symbol=2,
            )

        add_agent_to_observation(
            agent_size=agent_size,
            agent_x_position=agent_x_position,
            index=index,
            matrix_row=matrix_row,
            world_x_width=world_x_width,
            no_crashes=no_crashes,
        )

    # gymnasium expects an int64 numpy array
    return np.array(matrix, dtype=np.int64)


def add_agent_to_observation(
        agent_size, agent_x_position, index, matrix_row, world_x_width, no_crashes
):
    if index <= (2 * agent_size - 2):
        for index_agent in range(
                max(0, int(agent_x_position - agent_size + 1)),
                min(world_x_width + 2, int(agent_x_position + agent_size)),
        ):
            # means that the current number at the index of the agent can be 0 (nothing) or 2 (coin)
            # 0, 1, 2, 3
            if matrix_row[index_agent] == 0:
                matrix_row[index_agent] = 1
            elif matrix_row[index_agent] == -1 or matrix_row[index_agent] == -5:
                matrix_row[index_agent] = -10
            elif matrix_row[index_agent] == 2 or matrix_row[index_agent] == 3:
                if no_crashes:
                    matrix_row[index_agent] = 1
                else:
                    matrix_row[index_agent] = -10


def add_objects_to_observation(
        current_relevant_object_dict_list, current_y_position, matrix_row, symbol
):
    for obj in current_relevant_object_dict_list:
        # check if obj lays in current position
        if (
                obj["y"] - obj["size"] + 1
                <= current_y_position
                <= obj["y"] + obj["size"] - 1
        ):
            # if so, mark the x positions of the obstacles
            matrix_row[obj["x"] - obj["size"] + 1: obj["x"] + obj["size"]] = symbol


def add_drift_to_observation(
        observation_start_row, following_observation_size, drift_ranges, matrix
):
    # Per default, there is no drift on either side, just the walls
    matrix[:, 0] = -1
    matrix[:, -1] = -1

    for start, end, drift, is_visible, is_fake in drift_ranges:
        # Check whether either end of a drift is visible in the observation space, and the drift is visible
        if (
                not len(
                    range(
                        max(start, observation_start_row),
                        min(end, observation_start_row + following_observation_size) + 1,
                    ),
                )
                    == 0
        ) and is_visible:
            start_index = max(start - observation_start_row, 0)
            end_index = min(end - observation_start_row, following_observation_size)

            if drift < 0:
                matrix[start_index: end_index + 1, -1] = -5
            elif drift > 0:
                matrix[start_index: end_index + 1, 0] = -5


def add_funnels_to_observation(current_y_position, walls_dict, matrix_row):
    # depth of the funnel (1 at the first step, then 2, then 3, etc.)
    funnel_level = walls_dict[str(int(current_y_position))][0]
    if funnel_level > 0:
        matrix_row[1: funnel_level + 1] = -1
        matrix_row[-funnel_level - 1: -1] = -1


def find_visible_objects(
        following_observation_size: int,
        object_dict_list: List[Dict[str, int]],
        agent_y_position: int,
        agent_size: int = 1,
):
    # get relevant obstacles
    relevant_object_dict_list = []
    # check which obstacles are in the current observation space
    for obj in object_dict_list:
        if (
                agent_y_position + following_observation_size >= obj["y"] - obj["size"] + 1
                and obj["y"] + obj["size"] - 1 >= agent_y_position - agent_size + 1
        ):
            relevant_object_dict_list.append(obj)

    return relevant_object_dict_list


def calculate_gaussian_reward(state, collected_objects: list[dict], agent_size: int, task_type: str,
                              current_reward_function: str, x_position_of_agent: int, y_position_of_agent: int,
                              object_dict_list: list[dict] = None, no_crashes: bool = True) -> \
        tuple[int, list[dict]]:
    current_reward_gaussian = 0
    # remove agent from state
    blurred_state = copy.deepcopy(state)
    blurred_state[blurred_state == 1] = 0
    # handle drift as wall tiles
    blurred_state[blurred_state == -5] = -1
    # if no crashes in walls (and obstacles) are possible, remove walls from state
    if no_crashes:
        blurred_state[blurred_state == -1] = 0
    # replace walls (-1)(maybe already removed through no crashes) with the highest value (255)
    blurred_state[blurred_state == -1] = 255

    range_of_agent = range(-(agent_size - 1), agent_size)

    if len(collected_objects) > 0:
        for obj in collected_objects:
            # find positions where agent is on object --> only last row of agent is possible
            x_positions_of_agent = []
            y_positions_of_agent = []
            for index in range_of_agent:
                x_positions_of_agent.append(x_position_of_agent + index)
                y_positions_of_agent.append(y_position_of_agent + index)
            agent_positions = [(x_positions_of_agent[index_1], y_positions_of_agent[index_2]) for index_2 in
                               range(len(y_positions_of_agent)) for index_1 in
                               range(len(x_positions_of_agent))]

            range_of_object = range(
                -(math.floor(obj["size"] / 2)),
                (math.floor(obj["size"] / 2) + 1),
            )
            x_positions_of_object = []
            y_positions_of_object = []
            for index in range_of_object:
                x_positions_of_object.append(obj["x"] + index)
                y_positions_of_object.append(obj["y"] + index)
            object_positions = [(x_positions_of_object[index_1], y_positions_of_object[index_2]) for index_2 in
                                range(len(y_positions_of_object)) for index_1 in
                                range(len(x_positions_of_object))]

            positions_where_agent_is_on_object = list(
                set(agent_positions).intersection(object_positions)
            )
            # FIXME: what about multiple objects???

            if task_type == "coin":
                for pos in positions_where_agent_is_on_object:
                    # replace values of intersection of agent and coin with 2
                    blurred_state[pos[1] - y_position_of_agent + 1, pos[0]] = 2
            # obstacle
            else:
                for pos in positions_where_agent_is_on_object:
                    # replace values of intersection of agent and obstacle with 3
                    blurred_state[pos[1] - y_position_of_agent + 1, pos[0]] = 3

    if task_type == "coin":
        # replace nothing (0) with middle value (127)
        blurred_state[blurred_state == 0] = 127
        # replace coins (2) with the lowest value (0)
        blurred_state[blurred_state == 2] = 0
        # replace collect (-10) with the lowest value (0)
        # -10 here not possible for crashes, because it would already handled above
        blurred_state[blurred_state == -10] = 0
    else:
        # replace obstacles (3) with the highest value (255)
        blurred_state[blurred_state == 3] = 255
        # replace crash (-10) with the highest value (255)
        # -10 here not possible for crashes, because it would already handled above
        blurred_state[blurred_state == -10] = 255

    # apply gaussian filter (7x7)
    # gymnasium expects an int64 numpy array, but gaussian blur only works with int16
    blurred_state = cv2.GaussianBlur(blurred_state.astype(np.int16), (7, 7), 0)

    # get values of each pixel of current agent position
    values_of_agent_position = []
    # rows
    for i in range(2 * agent_size - 1):
        # columns
        if len(range_of_agent) == 0:
            values_of_agent_position.append(
                blurred_state[i][x_position_of_agent]
            )
        else:
            for j in range_of_agent:
                values_of_agent_position.append(
                    blurred_state[i][x_position_of_agent - j]
                )

    for value in values_of_agent_position:
        current_reward_gaussian += abs(value - 255)
    # normalize reward
    if task_type == "coin":
        # agent size 1 (1x1)
        # reward when no coin is near: 128 --> should be 0
        # lowest reward possible when collecting a coin: 146 --> should be 500
        if agent_size == 1:
            normalized_reward = ((current_reward_gaussian - 128) / 0.036) / 10
        # agent size 2 (3x3)
        # reward when no coin is near: 128*9=1152 --> should be 0
        # lowest reward possible when collecting a coin: 1309 --> should be 500
        # the function for this is: f(x) = 0.314x + 1152
        # we calculate the corresponding normalized reward x for the current reward f(x)
        # for reward of 10 when no coin is near:
        # normalized_reward = ((reward - 1152) / 0.314)/10
        elif agent_size == 2:
            normalized_reward = ((current_reward_gaussian - 1152) / 0.314) / 10
        # agent size 3 (5x5)
        # reward when no coin is near: 3200 -->  shouldbe 0
        # lowest reward possible when collecting a coin: 3373 --> should be 500
        elif agent_size == 3:
            normalized_reward = ((current_reward_gaussian - 3200) / 0.346) / 10
        # agent size 4 (7x7)
        # reward when no coin is near: 6272 --> should be 0
        # lowest reward possible when collecting a coin: 6445 --> should be 500
        elif agent_size == 4:
            normalized_reward = ((current_reward_gaussian - 6272) / 0.346) / 10
        # agent size 5 (9x9)
        # reward when no coin is near: 10368 --> should be 0
        # lowest reward possible when collecting a coin: 10541 --> should be 500
        elif agent_size == 5:
            normalized_reward = ((current_reward_gaussian - 10368) / 0.346) / 10
        # other sizes:
        # when an agent "collides" with the coin on just one corner the reward is always 127, 126, 126, 124,
        #                                                                                126, 124, 121, 116,
        #                                                                                126, 121, 111, 98,
        #                                                                                124, 116, 98, 75
        # in the colliding corner, while the other values in the agent position are 127
        else:
            reward_no_coin = agent_size * 128  # reward when no coin is near
            normalized_reward = ((current_reward_gaussian - reward_no_coin) / 0.346) / 10
    else:
        # agent size 1 (1x1)
        # biggest reward possible when near an obstacle: 231 --> should be 0
        # biggest reward possible when not near an obstacle: 255 --> should be 10
        if agent_size == 1:
            normalized_reward = (current_reward_gaussian - 231) / 2.4
        # agent size 2 (3x3)
        # we want the same distance as before 10 for successful step, 0 if near an obstacle (obstacle task)
        # the biggest reward possible when near an obstacle is 2219 --> this should be 0
        # the highest reward possible when not near an obstacle is 255*9=2295 --> this should be 10
        # the function for this is: f(x) = 7.6x + 2219
        # we calculate the corresponding normalized reward x for the current reward f(x)
        elif agent_size == 2:
            normalized_reward = (current_reward_gaussian - 2219) / 7.6
        # agent size 3 (5x5)
        # biggest reward possible when near an obstacle: 6303 --> should be 0
        # biggest reward possible when not near an obstacle: 255*25=6375 --> should be 10
        elif agent_size == 3:
            normalized_reward = (current_reward_gaussian - 6303) / 7.2
        # agent size 4 (7x7)
        # biggest reward possible when near an obstacle: 12423 --> should be 0
        # biggest reward possible when not near an obstacle: 255*49=12495 --> should be 10
        elif agent_size == 4:
            normalized_reward = (current_reward_gaussian - 12423) / 7.2
        # agent size 5 (9x9)
        # biggest reward possible when near an obstacle: 20583 --> should be 0
        # biggest reward possible when not near an obstacle: 255*81=20655 --> should be 10
        elif agent_size == 5:
            normalized_reward = (current_reward_gaussian - 20583) / 7.2
        # other sizes
        else:
            agent_size_matrix = (agent_size * 2 - 1) * (agent_size * 2 - 1)  # e.g. agent size = 4 --> 7x7=49
            smallest_reward_obstacle = 2223 + (agent_size_matrix - 9) * 255  # reward when near an obstacle
            normalized_reward = (current_reward_gaussian - smallest_reward_obstacle) / 7.2

    # boost (collect) or decrease (dodge) reward when collect objects
    if task_type == "coin":
        normalized_reward = normalized_reward + len(collected_objects) * 500
    else:
        normalized_reward = normalized_reward - len(collected_objects) * 500
    return int(normalized_reward), object_dict_list
