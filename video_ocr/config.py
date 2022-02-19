import click
from functools import wraps
_IS_CL = __name__ == "__main__"


def only_if_cl(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if _IS_CL:
            return f(*args, **kwargs)
    return wrapper

@only_if_cl
def error_log(text, *args, **kwargs):
    click.echo(click.style(text, fg="red"), err=True, *args, **kwargs)

@only_if_cl
def info_log(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)

