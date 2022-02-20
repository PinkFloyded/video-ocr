import click
from functools import wraps

IS_CL = False

def only_if_cl(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if IS_CL:
            return f(*args, **kwargs)
    return wrapper

@only_if_cl
def error_log(text, *args, **kwargs):
    click.echo(click.style(text, fg="red"), err=True, *args, **kwargs)

@only_if_cl
def info_log(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)

