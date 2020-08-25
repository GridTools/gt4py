"""Command line interface."""
import functools
import importlib
import pathlib
import sys
from typing import Any, Callable, Dict, KeysView, List, Optional, Tuple, Type, Union

import click
import tabulate

import gt4py
from gt4py import gtsimport
from gt4py.backend.base import Backend
from gt4py.lazy_stencil import LazyStencil


class BackendChoice(click.Choice):
    name = "backend"

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context],
    ) -> Type[Backend]:
        """Convert a CLI option argument to a backend."""
        name = super().convert(value, param, ctx)
        if not self.is_enabled(name):
            self.fail("Backend is not CLI-enabled.")
        return gt4py.backend.from_name(name)

    @staticmethod
    def get_backend_names() -> KeysView:
        return gt4py.backend.REGISTRY.keys()

    @staticmethod
    def is_enabled(backend_name: str) -> bool:
        """Check if a given backend is enabled for CLI."""
        backend_cls = gt4py.backend.from_name(backend_name)
        if hasattr(backend_cls, "generate_computation"):
            return True
        return False

    @classmethod
    def backend_table(cls) -> str:
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
        type_spec: Type,
        value: str,
        param: Optional[click.Parameter],
        ctx: Optional[click.Context],
    ) -> Any:
        if type_spec in self.converter_map:
            return self.converter_map[type_spec].convert(value, param, ctx)
        elif hasattr(type_spec, "convert"):
            return type_spec.convert(value, param, ctx)
        else:
            return type_spec(value)

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Tuple[str, Any]:
        backend = gt4py.backend.from_name("debug")
        if ctx:
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
        self.echo: Callable = self._noop
        self.secho: Callable = self._noop
        if not silent:
            self.echo = click.echo
            self.secho = click.secho
        self.error = functools.partial(click.echo, file=sys.stderr)

    @staticmethod
    def _noop(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> None:
        pass


def get_param_by_name(ctx: click.Context, name: str) -> click.Parameter:
    params = ctx.command.params
    by_name = {param.name: param for param in params}
    return by_name[name]


def write_computation_src(
    root_path: pathlib.Path, computation_src: Dict[str, Union[str, Dict]], reporter: Reporter
) -> None:
    for path_name, content in computation_src.items():
        if isinstance(content, Dict):
            root_path.joinpath(path_name).mkdir(exist_ok=True)
            write_computation_src(root_path / path_name, content, reporter)
        else:
            file_path = root_path / path_name
            reporter.echo(f"Writing source file: {file_path}")
            file_path.write_text(content)


def generate_stencils(
    stencils: List[LazyStencil],
    reporter: Reporter,
    *,
    output_path: pathlib.Path,
    backend: Optional[Backend] = None,
    build_options: Optional[Dict[str, Any]] = None,
) -> None:
    for proto_stencil in stencils:
        reporter.echo(f"Building stencil {proto_stencil.builder.options.name}")
        builder = proto_stencil.builder
        if backend:
            builder.with_backend(backend.name)
        if build_options:
            builder.with_changed_options(**build_options)
        builder.with_caching("nocaching", output_path=output_path)
        # how to tell mypy there can only be enabled backends here
        computation_src = builder.backend.generate_computation()  # type: ignore
        write_computation_src(builder.caching.root_path, computation_src, reporter)


def report_stencil_names(stencils: List[LazyStencil], reporter: Reporter) -> None:
    stencils_msg = "No stencils found."
    if stencils:
        stencil_names = ", ".join('"' + st.builder.options.name + '"' for st in stencils)
        stencils_msg = f"Found {len(stencils)} stencils: {stencil_names}."
    reporter.echo(stencils_msg)


@click.group()
def gtpyc():
    """
    GT4Py (GritTools for Python) stencil generator & compiler.

    This utility is currently only partially implemented.
    """


@gtpyc.command()
def list_backends():
    """List available backends."""
    reporter = Reporter(silent=False)
    reporter.echo(f"\n{BackendChoice.backend_table()}\n")
    return 0


@gtpyc.command()
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
@click.option("--silent", "-s", is_flag=True, help="suppress console output")
@click.argument(
    "input_path", required=True, type=click.Path(file_okay=True, dir_okay=True, exists=True)
)
def gen(backend, output_path, options, input_path, silent):
    """Generate stencils from gtscript modules or packages."""
    reporter = Reporter(silent=silent)
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    gtsimport.install(search_path=[input_path.parent])
    reporter.echo(f"reading input file {input_path}")
    input_module = importlib.import_module(input_path.stem.split(".")[0])
    reporter.echo(f"input file loaded as module {input_module}")
    build_options = dict(options)
    stencils = [
        v
        for k, v in input_module.__dict__.items()
        if k.startswith != "_" and isinstance(v, LazyStencil)
    ]
    report_stencil_names(stencils, reporter)
    generate_stencils(
        stencils, reporter, backend=backend, build_options=build_options, output_path=output_path
    )
