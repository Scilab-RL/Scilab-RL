"""Common aliases for type hints"""

from typing import NamedTuple

import torch as th

class ReplayBufferSamplesWithTestTrans(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    is_test_trans: th.Tensor
