import copy
import importlib
import shutil
import types
import tempfile
import pathlib

import click

import gt4py
from gt4py import gtsimport
from gt4py.build import BeginStage, BuildContext


def get_backend_options():
    from gt4py.backend import REGISTRY as backend_options

    return backend_options.keys()


def eval_arg_types(definition):
    """
    Call `eval()` on the function's type annotations after importing `Field`.

    This allows input modules to use string annotations of the type `arg: "Field[<dtype>]"`
    where <dtype> must be a name available in this function's scope.

    This is a poor quality version of what needs to be done in order to
    support input modules that do not import gt4py.
    """
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
    build_dir = pathlib.Path(tempfile.gettempdir())
    source = location.read_text()

    tmp_location = build_dir / location.name
    tmp_location.write_text(
        source.replace("## using-dsl: gtscript", "from gt4py.gtscript import *")
    )
    name = tmp_location.stem
    module_spec = importlib.util.spec_from_file_location(name=name, location=tmp_location)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def files_from_ctx(ctx):
    """
    Create a stencil object from a function definition.

    This does not use the `gtscript.stencil` function, as it needs access to
    the stencil id in order to retrieve the paths to the backend output files
    from the backend.

    Currently, this function creates and loads the python module in any case.
    To avoid that in cases where it is not necessary will require refactoring
    of the backend concept.
    """
    ir_stage = BeginStage(ctx).make_next()
    backend = ctx["backend"]
    stencil_obj = backend.generate(ctx["id"], ctx["ir"], ctx["definition"], ctx["options"])

    package_path = pathlib.Path(backend.get_stencil_package_path(ctx["id"]))
    return package_path


class BackendChoice(click.Choice):
    name = "backend"

    def convert(self, value, param, ctx):
        return gt4py.backend.from_name(super().convert(value, param, ctx))


class BindingsLanguage(click.ParamType):
    name = "bindings"

    def convert(self, value, param, ctx):
        backend = ctx.params["backend"]
        if value:
            if not backend.BINDINGS_LANGUAGES:
                self.fail(f"Backend {backend.name} can not generate language bindings!")
            if value not in backend.BINDINGS_LANGUAGES:
                suggestions = "\n".join(["\t* " + lang for lang in backend.BINDINGS_LANGUAGES])
                self.fail(
                    f"Backend {backend.name} can not generate bindings for {value}, try one of:\n{suggestions}"
                )
        return value


class BackendOption(click.ParamType):
    name = "otpion"

    converter_map = {bool: click.BOOL, int: click.INT, float: click.FLOAT, str: click.STRING}

    def _convert_value(self, type_spec, value, param, ctx):
        if type_spec in self.converter_map:
            return self.converter_map[type_spec].convert(value, param, ctx)
        elif hasattr(type_spec, "convert"):
            return type_spec.convert(value, param, ctx)
        else:
            return type_spec(value)

    def convert(self, value, param, ctx):
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


class JsonInput(click.ParamType):
    name = "json"

    def convert(self, value, param, ctx):
        if value:
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError as err:
                self.fail(f"invalid JSON string: {err}")
        return value


@click.command()
@click.option(
    "--backend",
    "-b",
    type=BackendChoice(get_backend_options()),
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
    "--bindings",
    type=BindingsLanguage(),
    help="Generate source code for language bindings. Which languages are available depends on the backend.",
)
@click.option(
    "--compile-bindings",
    "-C",
    is_flag=True,
    help="Compile bindings source code. Only has an effect in combination wih --bindings.",
)
@click.option(
    "--option",
    "-O",
    "options",
    multiple=True,
    type=BackendOption(),
    help="Backend flag (multiple allowed), format: -O key=value",
)
@click.option(
    "--externals",
    "-E",
    type=JsonInput(),
    help="JSON string describing externals overrides for all stencils.",
)
@click.argument("input_path", type=click.Path(file_okay=True, dir_okay=True, exists=True))
def gtpyc(input_path, backend, output_path, bindings, compile_bindings, options, externals):
    """
    GT4Py (GridTools for Python) stencil compiler.

    INPUT_PATH can be either a gtscript or python file or a python package.
    """

    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    finder = gtsimport.install(search_path=[input_path.parent])
    input_module = importlib.import_module(input_path.stem)
    build_options = dict(options)
    stencils = [
        v
        for k, v in input_module.__dict__.items()
        if k.startswith != "_" and isinstance(v, BuildContext)
    ]

    frontend = gt4py.frontend.from_name("gtscript")

    default_out_path_name = f"{input_path.stem}_out"

    # if output_path.exists() and output_path.name != default_out_path_name:
    #    output_path /= default_out_path_name

    if not output_path.exists():
        output_path.mkdir(mode=0o755)

    for ctx in stencils:
        ctx["output_path"] = output_path
        ctx["backend"] = backend
        ctx["frontend"] = frontend
        ctx["build_options"] = build_options or ctx["options"]
        ctx["externals"].update(externals or {})
        ctx["bindings"] = bindings
        builder = BeginStage(ctx).make()
        if not bindings and not backend.BINDINGS_LANGUAGES:
            stage = builder
            while not stage.is_final():
                stage = stage.make_next()
            mod_name = f"{ctx['name']}.py"
            mod_f = output_path / mod_name
            mod_f.write_text(ctx["src"][mod_name])
        elif compile_bindings:
            package_path = files_from_ctx(ctx)
            shutil.copytree(package_path, output_path / ctx["name"])
        else:
            ctx["pyext_module_name"] = "_" + ctx["name"]
            ctx["pyext_file_path"] = output_path / ctx["pyext_module_name"]
            stage = builder
            while not stage.is_final():
                stage = stage.make_next()
            for name in ctx["src"]:
                output_path.joinpath(name.replace("computation", ctx["name"])).write_text(
                    ctx["src"][name]
                )
            if bindings == "python":
                pybind_f = output_path / f"bindings.cpp"
                pybind_f.write_text(ctx["bindings_src"]["python"]["bindings.cpp"])
                mod_f = output_path / f'{ctx["name"]}.py'
                mod_f.write_text(ctx["bindings_src"]["python"][mod_f.name])
