"""Command line interface."""
import functools
import sys
import typing

import click
import tabulate

import gt4py


class BackendChoice(click.Choice):
    name = "backend"

    def convert(
        self,
        value: str,
        param: typing.Optional[click.Parameter],
        ctx: typing.Optional[click.Context],
    ):
        name = super().convert(value, param, ctx)
        if not self.is_enabled(name):
            self.fail("Backend is not CLI-enabled.")
        return gt4py.backend.from_name(name)

    @staticmethod
    def get_backend_names():
        return gt4py.backend.REGISTRY.keys()

    @staticmethod
    def is_enabled(backend_name: str):
        backend_cls = gt4py.backend.from_name(backend_name)
        if hasattr(backend_cls, "generate_computation"):
            return True
        return False

    @classmethod
    def backend_table(cls):
        headers = ["computation", "bindings", "CLI-enabled"]
        names = cls.get_backend_names()
        backends = [gt4py.backend.from_name(name) for name in names]
        comp_langs = [
            backend.languages["computation"] for backend in backends if backend.languages
        ]
        binding_langs = [
            ", ".join(backend.languages["bindings"]) for backend in backends if backend.languages
        ]
        enabled = [cls.is_enabled(name) and "Yes" or "No" for name in names]
        data = zip(names, comp_langs, binding_langs, enabled)
        return tabulate.tabulate(data, headers=headers)


class Reporter:
    """Wrapper around click echo functions or noops depending on the `silent` constructor param."""

    def __init__(self, silent: bool = False):
        self.echo: typing.Callable = self._noop
        self.secho: typing.Callable = self._noop
        if not silent:
            self.echo = click.echo
            self.secho = click.secho
        self.error = functools.partial(click.echo, file=sys.stderr)

    @staticmethod
    def _noop(*args, **kwargs):
        pass


def get_param_by_name(ctx: click.Context, name: str):
    params = ctx.command.params
    by_name = {param.name: param for param in params}
    return by_name[name]


@click.command()
@click.option(
    "--backend",
    "-b",
    type=BackendChoice(BackendChoice.get_backend_names()),
    help="Choose a backend",
    is_eager=True,
)
@click.option("--list-backends", is_flag=True, help="list backends and exit")
@click.option("--silent", "-s", is_flag=True, help="suppress console output")
@click.pass_context
@click.argument(
    "input_path", required=False, type=click.Path(file_okay=True, dir_okay=True, exists=True)
)
def gtpyc(ctx, backend, input_path, list_backends, silent):
    """
    GT4Py (GritTools for Python) stencil generator & compiler.

    This utility is currently only partially implemented.
    """
    reporter = Reporter(silent)
    if list_backends:
        reporter.echo(f"\n{BackendChoice.backend_table()}\n")
        return 0
    elif input_path is None:
        raise click.MissingParameter(ctx=ctx, param=get_param_by_name(ctx, "input_path"))
