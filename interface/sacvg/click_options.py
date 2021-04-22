import click
options = [
click.option('--verbose', type=bool, default=True),
click.option('--set_fut_ret_zero_if_done', type=int, default=1, help='Whether to set the future expected return to 0 if an episode is done when computing the TD q value.'),
]

def click_main(func):
    for option in reversed(options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs