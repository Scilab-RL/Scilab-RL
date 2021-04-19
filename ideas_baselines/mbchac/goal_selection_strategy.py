from enum import Enum


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """

    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    FUTURE2 = 1
    FUTURE3 = 2

    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 3
    # Select a goal that was achieved in the episode
    EPISODE = 4
    # randomly after the current step, in the same episode, but samples are drawn more towards the end of the episode
    RNDEND = 5
    RNDEND2 = 6
    RNDEND3 = 7

# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future": GoalSelectionStrategy.FUTURE,
    "future2": GoalSelectionStrategy.FUTURE2,
    "future3": GoalSelectionStrategy.FUTURE3,
    "final": GoalSelectionStrategy.FINAL,
    "episode": GoalSelectionStrategy.EPISODE,
    "rndend": GoalSelectionStrategy.RNDEND,
    "rndend2": GoalSelectionStrategy.RNDEND2,
    "rndend3": GoalSelectionStrategy.RNDEND2,
}
