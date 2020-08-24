"""Command line interface."""
import functools
import importlib
import pathlib
import sys
import typing

import click
import tabulate

import gt4py
from gt4py.lazy_stencil import LazyStencil
from gt4py import gtsimport


class BackendChoice(click.Choice):
    name = "backend"

    def convert(
        self,
        value: str,
        param: typing.Optional[click.Parameter],
        ctx: typing.Optional[click.Context],
    ):
        """Convert a CLI option argument to a backend."""
        name = super().convert(value, param, ctx)
        if not self.is_enabled(name):
            self.fail("Backend is not CLI-enabled.")
        return gt4py.backend.from_name(name)

    @staticmethod
    def get_backend_names():
        return gt4py.backend.REGISTRY.keys()

    @staticmethod
    def is_enabled(backend_name: str):
        """Check if a given backend is enabled for CLI."""
        backend_cls = gt4py.backend.from_name(backend_name)
        if hasattr(backend_cls, "generate_computation"):
            return True
        return False

    @classmethod
    def backend_table(cls):
        """Build a string with a table of backend compatibilities."""
        headers = ["computation", "bindings", "CLI-enabled"]
        names = cls.get_backend_names()
        backends = [gt4py.backend.from_name(name) for name in names]
        comp_langs = [
            backend.languages["computation"] if backend.languages else "?" for backend in backends
        ]
        binding_langs = [
            ", ".join(backend.languages["bindings"]) if backend.languages else "?"
            for backend in backends
        ]
        enabled = [cls.is_enabled(name) and "Yes" or "No" for name in names] + ["No"]
        data = zip(names, comp_langs, binding_langs, enabled)
        return tabulate.tabulate(data, headers=headers)


class BackendOption(click.ParamType):
    name = "otpion"

    converter_map = {bool: click.BOOL, int: click.INT, float: click.FLOAT, str: click.STRING}

    def _convert_value(
        self,
        type_spec: typing.Type,
        value: typing.Optional[str],
        param: click.Parameter,
        ctx: click.Context,
    ):
        if type_spec in self.converter_map:
            return self.converter_map[type_spec].convert(value, param, ctx)
        elif hasattr(type_spec, "convert"):
            return type_spec.convert(value, param, ctx)
        else:
            return type_spec(value)

    def convert(self, value: typing.Optional[str], param: click.Parameter, ctx: click.Context):
        backend = ctx.params["backend"]
        if value:
            name, value = value.split("=")
            if name.strip() not in backend.options:
                self.fail(f"Backend {backend.name} received unknown option: {name}!")
            try:
                value = self._convert_value(backend.options[name]["type"], value, param, ctx)
            except click.BadParameter as conversion_error:
                self.fail(f'Invalid value for backend option "{name}": {conversion_error.message}')
        return (name, value)


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
@click.option(
    "--output-path",
    "-o",
    default=".",
    type=click.Path(file_okay=False),
    help="output path for the compiled source.",
)
@click.option(
    "--option",
    "-O",
    "options",
    multiple=True,
    type=BackendOption(),
    help="Backend option (multiple allowed), format: -O key=value",
)
@click.option("--list-backends", is_flag=True, help="list backends and exit")
@click.option("--silent", "-s", is_flag=True, help="suppress console output")
@click.pass_context
@click.argument(
    "input_path", required=False, type=click.Path(file_okay=True, dir_okay=True, exists=True)
)
def gtpyc(ctx, backend, output_path, options, input_path, list_backends, silent):
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

    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    finder = gtsimport.install(search_path=[input_path.parent])
    reporter.echo("reading input file {input_path}")
    input_module = importlib.import_module(input_path.stem.split(".")[0])
    reporter.echo(f"input file loaded as module {input_module}")
    build_options = dict(options)
    stencils = [
        v
        for k, v in input_module.__dict__.items()
        if k.startswith != "_" and isinstance(v, LazyStencil)
    ]
    stencils_msg = "No stencils found."
    if stencils:
        stencil_names = ", ".join('"' + st.builder.options.name + '"' for st in stencils)
        stencils_msg = f"Found {len(stencils)} stencils: {stencil_names}."
    reporter.echo(stencils_msg)

    for proto_stencil in stencils:
        reporter.echo(f"Building stencil {proto_stencil.builder.options.name}")
        builder = proto_stencil.builder
        if backend:
            builder.with_backend(backend)
        builder.with_changed_options(**build_options)
        builder.with_caching("nocaching", output_path=output_path)
        computation_src = builder.backend.generate_computation()
        for file_name, file_content in computation_src.items():
            file_path = builder.caching.root_path / file_name
            reporter.echo(f"Writing source file: {file_path}")
            file_path.write_text(file_content)
