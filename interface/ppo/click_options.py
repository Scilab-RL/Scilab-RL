import click
options = [
click.option('--verbose', type=bool, default=False),
]

def click_main(func):
    for option in reversed(options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs