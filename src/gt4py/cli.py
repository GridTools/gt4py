import importlib
import shutil
import types

import click
import pathlib

import gt4py
from gt4py.backend import REGISTRY as backend_options


def eval_arg_types(definition):
    """
    Call `eval()` on the function's type annotations after importing `Field`.

    This allows input modules to use string annotations of the type `arg: "Field[<dtype>]"`
    where <dtype> must be a name available in this function's scope.

    This is a poor quality version of what needs to be done in order to
    support input modules that do not import gt4py.
    """
    from gt4py.gtscript import Field

    assert isinstance(definition, types.FunctionType)
    annotations = getattr(definition, "__annotations__", {})
    for arg, value in annotations.items():
        if isinstance(value, str):
            annotations[arg] = eval(value)
    return definition


def module_from_path(location: pathlib.Path):
    """
    Load a python module file residing in <location> as an imported module.

    This possibly duplicates functionality available elsewhere.
    """
    name = location.stem
    module_spec = importlib.util.spec_from_file_location(name=name, location=location)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def files_from_def(function_definition, frontend, backend):
    """
    Create a stencil object from a function definition.

    This does not use the `gtscript.stencil` function, as it needs access to
    the stencil id in order to retrieve the paths to the backend output files
    from the backend.

    Currently, this function creates and loads the python module in any case.
    To avoid that in cases where it is not necessary will require refactoring
    of the backend concept.
    """
    module = getattr(function_definition, "__module__", "")
    name = function_definition.__name__
    externals = {}
    qualified_name = f"{module}.{name}"
    build_options = gt4py.definitions.BuildOptions(name=name, module=module)
    build_options_id = backend.get_options_id(build_options)
    stencil_def = eval_arg_types(function_definition)
    stencil_id = frontend.get_stencil_id(qualified_name, stencil_def, externals, build_options_id)
    stencil_ir = frontend.generate(
        definition=function_definition, externals=externals, options=build_options
    )
    stencil_obj = backend.generate(stencil_id, stencil_ir, stencil_def, build_options)

    package_path = pathlib.Path(backend.get_stencil_package_path(stencil_id))
    return package_path


@click.command()
@click.option(
    "--backend",
    "-b",
    type=click.Choice(backend_options.keys()),
    help="Choose a backend",
    required=True,
)
@click.option(
    "--output-path",
    "-o",
    default=".",
    type=click.Path(file_okay=False),
    help="output path for the compiled source.",
)
@click.argument("input_file", type=click.Path(file_okay=True, dir_okay=False, exists=True))
def gtpyc(input_file, backend, output_path):
    """
    GT4Py (GridTools for Python) stencil compiler.
    """

    input_file = pathlib.Path(input_file)
    output_path = pathlib.Path(output_path)
    input_module = module_from_path(input_file)
    functions = [
        v
        for k, v in input_module.__dict__.items()
        if k.startswith != "_" and isinstance(v, types.FunctionType)
    ]

    frontend = gt4py.frontend.from_name("gtscript")
    backend = gt4py.backend.from_name(backend)

    default_out_path_name = f"{input_file.stem}_out"

    if output_path.exists() and output_path.name != default_out_path_name:
        output_path /= default_out_path_name
        output_path.mkdir(mode=755)

    for function in functions:
        package_path = files_from_def(function, frontend, backend)
        shutil.copytree(package_path, output_path / function.__name__)
