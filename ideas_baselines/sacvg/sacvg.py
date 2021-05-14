from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from ideas_baselines.sacvg.continuous_optimistic_critic import ContinuousOptimisticCritic
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.policies import register_policy
import gym
from stable_baselines3.common.preprocessing import get_action_dim

class SACOptiCriticPolicy(SACPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 **kwargs):
        if 'lower_q_limit' in kwargs:
            self.lower_q_limit = kwargs['lower_q_limit']
            del kwargs['lower_q_limit']
        else:
            self.lower_q_limit = None

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousOptimisticCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if self.lower_q_limit is not None:
            critic_kwargs['lower_q_limit'] = self.lower_q_limit
        return ContinuousOptimisticCritic(**critic_kwargs).to(self.device)



register_policy("MlpOptiCriticPolicy", SACOptiCriticPolicy)

class SACVG(SAC):
    """
    Soft Actor-Critic vis Variable Gamma (SACVG)
    Same as stable baselines version but uses a gamma that depends on whether a transition was a testing transition.
    Also, SACVG features an optimistic critic initialization.

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        # hindsight_sampling_done_if_success: int = 1,
        set_fut_ret_zero_if_done: int = 1,
        **kwargs):

        # self.hindsight_sampling_done_if_success = hindsight_sampling_done_if_success
        self.set_fut_ret_zero_if_done = set_fut_ret_zero_if_done
        if 'lower_q_limit' in kwargs:
            self.lower_q_limit = kwargs['lower_q_limit']
            del kwargs['lower_q_limit']
        else:
            self.lower_q_limit = None

        super(SACVG, self).__init__(
            **kwargs
        )
        if self.lower_q_limit is not None:
            self.policy_kwargs.update({'lower_q_limit': self.lower_q_limit})

    # def __init__(
    #     self,
    #     policy: Union[str, Type[SACPolicy]],
    #     env: Union[GymEnv, str],
    #     learning_rate: Union[float, Callable] = 3e-4,
    #     buffer_size: int = int(1e6),
    #     learning_starts: int = 100,
    #     batch_size: int = 256,
    #     tau: float = 0.005,
    #     gamma: float = 0.99,
    #     train_freq: int = 1,
    #     gradient_steps: int = 1,
    #     n_episodes_rollout: int = -1,
    #     action_noise: Optional[ActionNoise] = None,
    #     optimize_memory_usage: bool = False,
    #     ent_coef: Union[str, float] = "auto",
    #     target_update_interval: int = 1,
    #     target_entropy: Union[str, float] = "auto",
    #     use_sde: bool = False,
    #     sde_sample_freq: int = -1,
    #     use_sde_at_warmup: bool = False,
    #     tensorboard_log: Optional[str] = None,
    #     create_eval_env: bool = False,
    #     policy_kwargs: Dict[str, Any] = None,
    #     verbose: int = 0,
    #     seed: Optional[int] = None,
    #     device: Union[th.device, str] = "auto",
    #     _init_setup_model: bool = True,
    # ):
    #
    #     super(SAC, self).__init__(
    #         policy,
    #         env,
    #         SACOptiCriticPolicy, # Use SACOptiCriticPolicy, not SACPolicy
    #         learning_rate,
    #         buffer_size,
    #         learning_starts,
    #         batch_size,
    #         tau,
    #         gamma,
    #         train_freq,
    #         gradient_steps,
    #         n_episodes_rollout,
    #         action_noise,
    #         policy_kwargs=policy_kwargs,
    #         tensorboard_log=tensorboard_log,
    #         verbose=verbose,
    #         device=device,
    #         create_eval_env=create_eval_env,
    #         seed=seed,
    #         use_sde=use_sde,
    #         sde_sample_freq=sde_sample_freq,
    #         use_sde_at_warmup=use_sde_at_warmup,
    #         optimize_memory_usage=optimize_memory_usage,
    #     )
    #
    #     self.target_entropy = target_entropy
    #     self.log_ent_coef = None  # type: Optional[th.Tensor]
    #     # Entropy coefficient / Entropy temperature
    #     # Inverse of the reward scale
    #     self.ent_coef = ent_coef
    #     self.target_update_interval = target_update_interval
    #     self.ent_coef_optimizer = None
    #
    #     if _init_setup_model:
    #         self._setup_model()

    # def _setup_model(self) -> None:
    #     # Setup model as before
    #     super()._setup_model()
    #     # But overwrite critic with optimisitcally initialized critic
    #     self.policy = self.policy_class(
    #         self.observation_space,
    #         self.action_space,
    #         self.lr_schedule,
    #         **self.policy_kwargs  # pytype:disable=not-instantiable
    #     )
    #     self.policy = self.policy.to(self.device)

        # self.policy.critic = # TBD


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                # add entropy term
                target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                if hasattr(replay_data, 'is_test_trans'):
                    gamma = self.gamma * (1 - replay_data.is_test_trans)
                else:
                    gamma = self.gamma
                if self.set_fut_ret_zero_if_done:
                    q_backup = replay_data.rewards + (1 - replay_data.dones) * gamma * target_q
                else:
                    q_backup = replay_data.rewards + gamma * target_q

            # Get current Q estimates for each critic network
            # using action from the replay buffer
            current_q_estimates = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


