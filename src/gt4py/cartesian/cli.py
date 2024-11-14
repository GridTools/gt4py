# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Command line interface."""

import functools
import importlib
import pathlib
import sys
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    KeysView,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import click
import tabulate

from gt4py import cartesian as gt4pyc
from gt4py.cartesian import gtscript_imports
from gt4py.cartesian.backend.base import CLIBackendMixin
from gt4py.cartesian.lazy_stencil import LazyStencil


class BackendChoice(click.Choice):
    """
    Backend commandline option type.

    Converts from name to backend class.

    Example
    -------
    .. code-block: bash

        $ cmd --backend="numpy"

    gets converted to :py:class:`gt4pyc.backend.GTCNumpyBackend`.
    """

    name = "backend"

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Type[CLIBackendMixin]:
        """Convert a CLI option argument to a backend."""
        name = super().convert(value, param, ctx)
        backend_cls = self.enabled_backend_cls_from_name(name)
        if backend_cls is None:
            self.fail("Backend is not CLI-enabled.")
        assert backend_cls is not None
        return backend_cls

    @staticmethod
    def get_backend_names() -> KeysView:
        return gt4pyc.backend.REGISTRY.keys()

    @staticmethod
    def enabled_backend_cls_from_name(backend_name: str) -> Optional[Type[CLIBackendMixin]]:
        """Check if a given backend is enabled for CLI."""
        backend_cls = gt4pyc.backend.from_name(backend_name)
        if backend_cls is None or not issubclass(backend_cls, CLIBackendMixin):
            return None
        return backend_cls

    @classmethod
    def backend_table(cls) -> str:
        """Build a string with a table of backend compatibilities."""
        headers = ["computation", "bindings", "CLI-enabled"]
        names = cls.get_backend_names()
        backends = [cls.enabled_backend_cls_from_name(name) for name in names]
        comp_langs = [
            backend.languages["computation"] if backend and backend.languages else "?"
            for backend in backends
        ]
        binding_langs = [
            ", ".join(backend.languages["bindings"]) if backend and backend.languages else "?"
            for backend in backends
        ]
        enabled = [backend is not None and "Yes" or "No" for backend in backends]
        data = zip(names, comp_langs, binding_langs, enabled)
        return tabulate.tabulate(data, headers=headers)


class BackendOption(click.ParamType):
    """
    Backend specific build options for commandline usage.

    convert from ``"key=value"`` strings to ``(key, value)`` tuples, where ``value`` is
    converted to the type declared for the chosen backend (looked up in the context).
    """

    name = "option"

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

    def _try_split(self, value: str) -> Tuple[str, str]:
        """Be helpful in case of formatting error."""
        try:
            name, value = value.split("=")
            if not name:
                raise ValueError("name can not be empty")
            if not value:
                raise ValueError("value can not be empty")
            return name, value
        except ValueError:
            self.fail('Invalid backend option format: must be "<name>=<value>"')
            return ("", "")

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Tuple[str, Any]:
        backend = ctx.params["backend"] if ctx else gt4pyc.backend.from_name("numpy")
        assert isinstance(backend, type)
        assert issubclass(backend, gt4pyc.backend.Backend)
        name, value = self._try_split(value)
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
        self.echo: Callable = click.echo
        self.secho: Callable = click.secho
        if silent:
            self.echo = self._noop
            self.secho = self._noop
        self.error = functools.partial(click.echo, file=sys.stderr)

    @staticmethod
    def _noop(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> None:
        pass


def get_param_by_name(ctx: click.Context, name: str) -> click.Parameter:
    params = ctx.command.params
    by_name = {param.name: param for param in params}
    return by_name[name]


class GTScriptBuilder:
    """
    Generate stencil source code from a GTScript module.

    Parameters
    ----------
    input_path :
        path (string or Pathlike) to the GTScript module.

    output_path :
        path (string or Pathlike) to where the generated source files should be written.

    backend :
        class of the backend that should be used.

    silent :
        silence all reporting to stdout if True

    """

    def __init__(
        self,
        input_path: Union[str, pathlib.Path],
        *,
        output_path: Union[str, pathlib.Path],
        backend: Type[CLIBackendMixin],
        silent: bool = False,
    ):
        self.reporter = Reporter(silent)
        self.input_module = self.import_input_module(pathlib.Path(input_path))
        self.output_path = pathlib.Path(output_path)
        self.backend_cls = backend

    def import_input_module(self, input_path: pathlib.Path) -> ModuleType:
        input_module = None
        with gtscript_imports.enabled(search_path=[input_path.parent]):
            self.reporter.echo(f"reading input file {input_path}")
            input_module = importlib.import_module(input_path.stem.split(".")[0])
            self.reporter.echo(f"input file loaded as module {input_module}")
        return input_module

    def iterate_stencils(self) -> Generator[LazyStencil, None, None]:
        return (
            v
            for k, v in self.input_module.__dict__.items()
            if k.startswith != "_" and isinstance(v, LazyStencil)
        )

    def write_computation_src(
        self, root_path: pathlib.Path, computation_src: Dict[str, Union[str, Dict]]
    ) -> None:
        for path_name, content in computation_src.items():
            if isinstance(content, Dict):
                root_path.joinpath(path_name).mkdir(exist_ok=True)
                self.write_computation_src(root_path / path_name, content)
            else:
                file_path = root_path / path_name
                self.reporter.echo(f"Writing source file: {file_path}")
                file_path.write_text(content)

    def generate_stencils(self, build_options: Optional[Dict[str, Any]] = None) -> None:
        for proto_stencil in self.iterate_stencils():
            self.reporter.echo(f"Building stencil {proto_stencil.builder.options.name}")
            builder = proto_stencil.builder.with_backend(self.backend_cls.name)
            if build_options:
                builder.with_changed_options(impl_opts=build_options)
            builder.with_caching("nocaching", output_path=self.output_path)
            computation_src = builder.generate_computation()
            self.write_computation_src(builder.caching.root_path, computation_src)

    def report_stencil_names(self) -> None:
        stencils = list(self.iterate_stencils())
        stencils_msg = "No stencils found."
        if stencils:
            stencil_names = ", ".join('"' + st.builder.options.name + '"' for st in stencils)
            stencils_msg = f"Found {len(stencils)} stencils: {stencil_names}."
        self.reporter.echo(stencils_msg)


@click.group()
def gtpyc() -> None:
    """
    GT4Py (GridTools for Python) stencil generator & compiler.

    This utility is currently only partially implemented.
    """


@gtpyc.command()
def list_backends() -> None:
    """List available backends."""
    reporter = Reporter(silent=False)
    reporter.echo(f"\n{BackendChoice.backend_table()}\n")


@gtpyc.command()
@click.option(
    "--backend",
    "-b",
    type=BackendChoice(cast(Sequence[str], BackendChoice.get_backend_names())),
    required=True,
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
def gen(
    backend: Type[CLIBackendMixin],
    output_path: str,
    options: Dict[str, Any],
    input_path: str,
    silent: bool,
) -> None:
    """Generate stencils from gtscript modules or packages."""
    GTScriptBuilder(
        input_path=input_path, output_path=output_path, backend=backend, silent=silent
    ).generate_stencils(build_options=dict(options))
