import copy
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


class BuildError(Exception):
    pass


class BuildContext:
    def __init__(self, definition, data_dict):
        self._data = data_dict
        self["definition"] = eval_arg_types(definition)
        self["module"] = getattr(definition, "__module__", "")
        self["name"] = definition.__name__
        self["externals"] = {}
        self["qualified_name"] = f"{self['module']}.{self['name']}"
        self["build_info"] = {}
        self["options"] = gt4py.definitions.BuildOptions(
            name=self["name"], module=self["module"], build_info=self["build_info"]
        )
        self._data.update(data_dict)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def externals(self):
        return self._data.get("externals", {})

    @property
    def build_info(self):
        return self._data.get("build_info", {})

    @property
    def backend(self):
        if "backend" not in self._data:
            raise BuildError("Backend not set.")
        return self._data["backend"]

    @backend.setter
    def backend(self, value):
        if "backend" in self._data:
            raise BuildError("Can not change backend, create new build context instead.")
        self._data["backend"] = value

    @property
    def frontend(self):
        if "frontend" not in self._data:
            raise BuildError("Frontend not set.")
        return self._data["frontend"]

    @frontend.setter
    def frontend(self, value):
        if "frontend" in self._data:
            raise BuildError("Can not change frontend, create new build context instead.")
        self._data["frontend"] = value

    @property
    def options(self):
        if "options" not in self._data:
            raise BuildError("Build options not set.")
        return self._data["options"]

    @options.setter
    def options(self, value):
        self._data["options"] = value

    @property
    def options_id(self):
        if "options_id" not in self._data:
            self._data["options_id"] = self.backend.get_options_id(self.options)
        return self._data["options_id"]

    def clone(self):
        return copy.deepcopy(self)


def backend_has_ext(backend):
    if hasattr(backend, "PYEXT_GENERATOR_CLASS"):
        return True
    return False


class BuildWrapper:
    def __init__(self, definition, *, frontend, backend, build_options):
        self.definition = definition
        self.ctx = BuildContext(definition, build_options)
        self.ctx["frontend"] = frontend
        self.ctx["backend"] = backend

    def make_ir(self):
        ctx = self.ctx.clone()
        ctx["id"] = ctx["frontend"].get_stencil_id(
            ctx["qualified_name"], ctx["definition"], ctx["externals"], ctx["options"]
        )
        ctx["ir"] = ctx["frontend"].generate(
            definition=ctx["definition"], externals=ctx["externals"], options=ctx["options"]
        )
        return IRWrapper(ctx["ir"], ctx=ctx)


class IRWrapper:
    def __init__(self, stencil_ir, *, ctx):
        self.stencil_ir = stencil_ir
        self.ctx = ctx

    def make_iir(self):
        ctx = self.ctx.clone()
        ctx["backend"]._check_options(ctx["options"])
        ctx["iir"] = gt4py.analysis.transform(self.stencil_ir, ctx["options"])
        if backend_has_ext(ctx["backend"]):
            return self._iir_with_pyext(ctx["iir"], ctx)
        return self._iir_no_pyext(ctx["iir"], ctx)

    def _iir_with_pyext(self, stencil_iir, ctx):
        return IIRWrapperWithPyext(stencil_iir, ctx=ctx)

    def _iir_no_pyext(self, stencil_iir, ctx):
        return IIRWrapperNoPyext(stencil_iir, ctx=ctx)


class IIRWrapper:
    def __init__(self, stencil_ir, *, ctx):
        self.stencil_iir = stencil_ir
        self.ctx = ctx


class IIRWrapperNoPyext(IIRWrapper):
    def make_module_src(self):
        ctx = self.ctx.clone()
        generator_options = copy.deepcopy(ctx["options"].as_dict())
        generator = ctx["backend"].GENERATOR_CLASS(ctx["backend"], options=generator_options)
        ctx["module_src"] = generator(ctx["id"], self.stencil_iir)
        return ModuleSrcWrapper(ctx["module_src"], ctx=ctx)


class IIRWrapperWithPyext(IIRWrapperNoPyext):
    def make_ext_sources(self):
        ctx = self.ctx.clone()
        backend = ctx["backend"]
        generator = backend.PYEXT_GENERATOR_CLASS(
            backend.get_pyext_class_name(ctx["id"]),
            backend.get_pyext_module_name(ctx["id"]),
            backend._CPU_ARCHITECTURE,
            ctx["options"],
        )
        ctx["pyext_src"] = generator(self.stencil_iir)
        return PyextSrcWrapper(ctx["pyext_src"], ctx=ctx)


class PyextSrcWrapper:
    def __init__(self, pyext_src, *, ctx):
        self.pyext_src = pyext_src
        self.ctx = ctx

    def make_module_src(self, *, ext_module_name, ext_module_path):
        ctx = self.ctx.clone()
        ctx["generator_options"] = ctx["options"].as_dict()
        ctx["generator_options"]["pyext_module_name"] = ext_module_name
        ctx["generator_options"]["pyext_file_path"] = ext_module_path
        generator = ctx["backend"].GENERATOR_CLASS(
            ctx["backend"], options=ctx["generator_options"]
        )
        ctx["module_src"] = generator(ctx["id"], ctx["iir"])
        return ModuleSrcWrapper(ctx["module_src"], ctx=ctx)

    @property
    def impl(self):
        return self.pyext_src["computation.src"]

    @property
    def header(self):
        return self.pyext_src["computation.hpp"]

    @property
    def bindings(self):
        return self.pyext_src["bindings.cpp"]

    @property
    def impl_ext(self):
        return "." + self.ctx["backend"].SRC_EXTENSION


class ModuleSrcWrapper:
    def __init__(self, module_src, *, ctx):
        self.module_src = module_src
        self.ctx = ctx

    @property
    def module(self):
        return self.module_src


class BackendChoice(click.Choice):
    name = "backend"

    def convert(self, value, param, ctx):
        return gt4py.backend.from_name(super().convert(value, param, ctx))


STAGE_CHOICES = {
    "nopy": "generate only backend specific source code",
    "withpy": "generate backend specific code and python bindings",
}


def verify_output_spec(ctx, param, value):
    backend = ctx.params["backend"]
    choices = STAGE_CHOICES
    if value:
        if not backend_has_ext(backend):
            raise click.BadParameter(f"backend {backend} does not allow source-only options")
    return value


@click.command()
@click.option(
    "--backend",
    "-b",
    type=BackendChoice(backend_options.keys()),
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
    "--src-only",
    callback=verify_output_spec,
    type=click.Choice(STAGE_CHOICES),
    help="Generate only source code with or without python bindings. Only available if backend supports it.",
)
@click.option(
    "--option",
    "-O",
    "options",
    nargs=2,
    multiple=True,
    type=(str, bool),
    help="Backend flag (multiple allowed), format: -O key value",
)
@click.argument("input_file", type=click.Path(file_okay=True, dir_okay=False, exists=True))
def gtpyc(input_file, backend, output_path, src_only, options):
    """
    GT4Py (GridTools for Python) stencil compiler.
    """

    input_file = pathlib.Path(input_file)
    output_path = pathlib.Path(output_path)
    input_module = module_from_path(input_file)
    build_options = dict(options)
    functions = [
        v
        for k, v in input_module.__dict__.items()
        if k.startswith != "_" and isinstance(v, types.FunctionType)
    ]

    frontend = gt4py.frontend.from_name("gtscript")

    default_out_path_name = f"{input_file.stem}_out"

    # if output_path.exists() and output_path.name != default_out_path_name:
    #    output_path /= default_out_path_name

    if not output_path.exists():
        output_path.mkdir(mode=0o755)

    for function in functions:
        if not src_only:
            package_path = files_from_def(function, frontend, backend)
            shutil.copytree(package_path, output_path / function.__name__)
        else:
            stage_iir = (
                BuildWrapper(
                    function, frontend=frontend, backend=backend, build_options=build_options
                )
                .make_ir()
                .make_iir()
            )
            ext_src = stage_iir.make_ext_sources()
            header_f = output_path / f'{ext_src.ctx["name"]}.hpp'
            impl_f = output_path / f'{ext_src.ctx["name"]}{ext_src.impl_ext}'
            header_f.write_text(ext_src.header)
            impl_f.write_text(ext_src.impl)
            if src_only == "withpy":
                pybind_f = output_path / f"bindings.cpp"
                pybind_f.write_text(ext_src.bindings)
                ext_mod_name = "_" + ext_src.ctx["name"]
                mod_src = ext_src.make_module_src(
                    ext_module_name=ext_mod_name, ext_module_path=output_path / ext_mod_name
                )
                mod_f = output_path / f'{ext_src.ctx["name"]}.py'
                mod_f.write_text(mod_src.module)
