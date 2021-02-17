import click
import importlib

options = [
click.option('--model_class', type=str, default='sac'),
click.option('--goal_selection_strategy', type=click.Choice(['future', 'final', 'episode']), default='future'),
click.option('--n_sampled_goal', type=int, default=4),
click.option('--online_sampling', type=bool, default=True),
click.option('--verbose', type=bool, default=True),
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