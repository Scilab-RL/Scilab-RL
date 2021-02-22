import click
import importlib

# default name: 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1'

_global_options = [
click.option('--env', type=str, default='AntFourRoomsEnv-v0', help='the name of the OpenAI Gym environment that you want to train on. E.g. TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-2-v1, AntFourRoomsEnv-v0'),
click.option('--algorithm', default='sac', help='the name of the algorithm to be used',
             type=click.Choice(['td3', 'sac', 'dqn', 'ddpg', 'her', 'mbchac'])),
# click.option('--action_steps', type=int, default=0, help='The total number of action steps. 0 indicates using the max_episode_steps property of the environment. Any non-zero numbers will overwrite the max_episode_steps property'),
click.option('--base_logdir', type=str, default='data', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/'),
# click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)'),
click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code. 0 means random seed'),
click.option('--restore_policy', type=str, default=None, help='The pretrained policy file to start with to avoid learning from scratch again. Useful for interrupting and restoring training sessions.'),
click.option('--rollout_batch_size', type=int, default=1, help='The number of simultaneous rollouts.'),
click.option('--eval_after_n_actions', type=int, default=5000, help='The number of training actions after which to evaluate the policy.'),
click.option('--n_epochs', type=int, default=30, help='the max. number of training epochs to run. One epoch consists of \'eval_after_n_steps\' actions '),
click.option('--n_test_rollouts', type=int, default=25, help='The number of testing rollouts.'),
# click.option('--render_train', type=int, default=0, help='Whether or not to render the rollout execution during training. Currently not implemented.'),
click.option('--render_test', type=int, default=0, help='Whether or not to render the rollout execution during testing.'),
click.option('--max_try_idx', type=int, default=199, help='Max. number of tries for this training config.'),
click.option('--try_start_idx', type=int, default=100, help='Index for first try.'),
click.option('--early_stop_threshold', type=float, default=0.99, help='The early stopping threshold.'),
click.option('--early_stop_data_column', type=str, default='test/success_rate', help='The data column on which early stopping is based.'),
click.option('--info', type=str, default='', help='A command line comment that will be integrated in the folder where the results are stored. Useful for debugging and addressing temporary changes to the code..'),
# click.option('--graph', type=int, default=1, help='Whether or not to create the online graph in environments that support it.'),
click.option('--tensorboard', type=int, default=2, help='max. number of tensorboard instances allowed at the same time. Will be determined by number of open ports, starting at port 6006'),
click.option('--plot_eval_cols', type=str, default='test/success_rate,test/mean_reward', help='Data to plot for evaluation. Strings separated by comma.'),
# click.option('--reward_type', type=str, default='sparse', help='the reward type, dense or sparse')
]


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
def get_algorithm_click(ctx, **kwargs):
    policy_linker = importlib.import_module('interface.' + kwargs['algorithm'] + ".click_options", package=__package__)
    policy_args = ctx.forward(policy_linker.get_click_option)
    return policy_args

def import_creator(library_path):
    config = importlib.import_module('interface.' + library_path + ".config", package=__package__)
    return config


def click_main(func):
    for option in reversed(_global_options):
        func = option(func)
    return func