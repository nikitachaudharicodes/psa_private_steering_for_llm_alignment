"""CLI interface for iclr2025_psa project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""
from .base import BaseClass
from .subpackage import SubPackageClass

def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m iclr2025_psa` and `$ iclr2025_psa `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    bc = BaseClass("test")
    print(f"This will do something: {bc.something()}")

    spc = SubPackageClass("test")
    print(f"This will do something else: {spc.something()}")
