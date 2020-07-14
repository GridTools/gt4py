"""Command line interface."""
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
        if hasattr(backend_cls, "generate_computation_src"):
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


@click.command()
@click.option(
    "--backend",
    "-b",
    type=BackendChoice(BackendChoice.get_backend_names()),
    help="Choose a backend",
    is_eager=True,
)
@click.option("--list-backends", is_flag=True, help="list backends and exit")
def gtpyc(backend, list_backends):
    """
    GT4Py (GritTools for Python) stencil generator & compiler.

    This utility is currently only partially implemented.
    """
    if list_backends:
        click.echo("")
        click.echo(BackendChoice.backend_table())
        click.echo("")
        return 0
