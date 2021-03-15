import click
import importlib

options = [
click.option('--model_classes', type=str, default='sac,sac'),
click.option('--goal_selection_strategy', type=click.Choice(['future', 'final', 'episode']), default='future'),
click.option('--n_sampled_goal', type=int, default=4),
click.option('--train_freq', type=int, default=0, help='number of steps in each layer after which to train. 0 sets the training frequency to once per episode.'),
click.option('--n_train_batches', type=int, default=0, help='The number of training batches per episode. 0 sets training batches to number of actions executed since last training in each layer, which effectively means that n_train_batches=n_train_freq.'),

click.option('--learning_starts', type=int, default=100, help='The number of transitions in each layer required to start NN training.'),

# click.option('--batch_size', type=int, default=1024, help='The number of state transitions processed during network training.'),
# click.option('--n_train_batches', type=int, default=40, help='The number of batches to train the actor-critic .'),
# click.option('--buffer_size', type=str, default="500,500", help='The number of episodes to store in each level\'s replay buffer.'),
# misc
# click.option('--num_threads', type=int, default=1, help='Number of threads used for intraop parallelism on CPU'),
# click.option('--verbose', type=bool, default=False),

click.option('--render_train', type=click.Choice(['record', 'display', 'none']), default='none', help='Whether and how to render the rollout execution during training. \'record\' is for video, \'display\' for direct visualization.'),
click.option('--render_test', type=click.Choice(['record', 'display', 'none']), default='none', help='Whether and how to render the rollout execution during testing. \'record\' is for video, \'display\' for direct visualization.'),

# HAC
# click.option('--target_networks', type=int, default=1),
# click.option('--tau', type=float, default=0.005, help='Tau to update target networks'),
# click.option('--q_lr', type=float, default=0.001, help='Critic learning rate'),
# click.option('--q_hidden_size', type=int, default=64, help='Hidden size used for the critic network'),
# click.option('--mu_lr', type=float, default=0.001, help='Actor learning rate'),
# click.option('--mu_hidden_size', type=int, default=64, help='Hidden size used for the actor network'),
# click.option('--continuous_subgoals', type=bool, default=False, help='Whether to determine new subgoals with each step.'), # TODO: CURRENTLY NOT IMPLEMENTED!
click.option('--time_scales', type=str, default='5,_', help='Steps per level from lowest to highest, separated by comma. There must be one \'_\' character in the list to indicates the layer where the time scale is determined from the environment\'s predefined steps.'),

click.option('--subgoal_test_perc', type=float, default=0.3, help='The percentage of subgoals to test.'),
# click.option('--level_types', type=str, default='hac,hac', help='Layers to be used'),
# click.option('--simulate_level', default='0,0', help='Specifiy for which level to add simulated transitions to the replay buffer'),
# click.option('--halftime', type=int, default=10000, help='Denotes the number of training episodes where the probability of simulating is 50%'),
#
# # click.option('--n_pre_episodes', type=int, default=30, help='Number of finished episodes before training actor-critic and dynamics model'), # This is realized with learning_begins
# click.option('--learning_begins', type=int, default=30, help='Number of finished episodes before training actor-critic and dynamics model'), # This is realized with learning_begins
# click.option('--random_action_perc', type=float, default=0.3, help='Percentage of taking random actions'),
# click.option('--atomic_noise', type=float, default=0.2, help='Exploration noise added to atomic actions'),
# click.option('--subgoal_noise', type=float, default=0.2, help='Exploration noise added to subgoal actions'),
# # click.option('--normalize', type=int, default=1 , help='Normalize states and goals for training the actor-critic, and also actions for the dynamics model'),
#
# # dynamics model
# click.option('--eta', type=str, default='0.5,0.5', help='Level-wise reward fraction r_i = (r_intrinsic * eta + (1-eta) * r_env)'),
# # click.option('--n_layer', default=2, type=int),
# click.option('--dm_ensemble', type=int, default=0, help='Size of dynamics model ensemble'),
# click.option('--dm_train_freq', type=int, default=20, help="Frequency of training dynamics model: every n training episodes"),
# click.option('--dm_hidden_size', type=int, default=128),
# click.option('--dm_lr', type=float, default=0.001, help='Learning rate to train the dynamics model'),
# click.option('--dm_updates', type=int, default=20, help='Number of gradient updates'),
# click.option('--dm_type', default='mlp', type=click.Choice(['mlp', 'rnn'])),
# click.option('--dm_buffer_size', type=int, default=1000, help='Number of trajectories to store'),
# click.option('--err_buffer_size', type=int, default=1e+7, help="Size of error buffer"),
# click.option('--dm_batch_size', type=int, default=1024, help='Number of trajectories or transitions to sample for training'),
]

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
def get_model_class_click(ctx, **kwargs):
    policy_linker = importlib.import_module('interface.' + kwargs['model_class'] + ".click_options", package=__package__)
    policy_args = ctx.forward(policy_linker.get_click_option)
    return policy_args

def click_main(func):
    for option in reversed(options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs