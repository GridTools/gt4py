import copy
import importlib
import shutil
import types
import tempfile
import pathlib

import click

import gt4py
from gt4py.backend import REGISTRY as backend_options
from gt4py import gtsimport


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
    ir_stage = BuildWrapper.from_context(ctx).make_ir()
    ctx = ir_stage.ctx
    backend = ctx["backend"]
    stencil_obj = backend.generate(
        ctx["id"], ir_stage.stencil_ir, ctx["definition"], ctx["options"]
    )

    package_path = pathlib.Path(backend.get_stencil_package_path(ctx["id"]))
    return package_path


class BuildError(Exception):
    pass


class BuildContext:
    def __init__(self, definition, data_dict):
        self._data = data_dict
        self["definition"] = eval_arg_types(definition)
        self["module"] = getattr(definition, "__module__", "")
        self["name"] = definition.__name__
        self["externals"] = self._data.get("externals", {})
        self["qualified_name"] = f"{self['module']}.{self['name']}"
        self["build_info"] = self._data.get("build_info", {})
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

    def update(self, data):
        self._data.update(data)


def backend_has_ext(backend):
    if hasattr(backend, "PYEXT_GENERATOR_CLASS"):
        return True
    return False


class BuildWrapper:
    def __init__(self, definition, *, frontend, backend, build_options, ctx=None):
        self.definition = definition
        self.ctx = ctx or BuildContext(definition, build_options)
        self.ctx["frontend"] = frontend
        self.ctx["backend"] = backend

    def make_ir(self):
        ctx = self.ctx.clone()
        ctx["id"] = ctx["frontend"].get_stencil_id(
            ctx["qualified_name"], ctx["definition"], ctx["externals"].copy(), ctx["options"]
        )
        ctx["ir"] = ctx["frontend"].generate(
            definition=ctx["definition"], externals=ctx["externals"].copy(), options=ctx["options"]
        )
        return IRWrapper(ctx["ir"], ctx=ctx)

    def make_next(self):
        return self.make_ir()

    @classmethod
    def from_context(cls, ctx, **kwargs):
        ctx = ctx.clone()
        ctx.externals.update(kwargs.pop("externals", {}))
        ctx.update(kwargs)
        return cls(
            ctx["definition"],
            frontend=ctx["frontend"],
            backend=ctx["backend"],
            build_options=ctx["build_options"],
            ctx=ctx,
        )


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

    def make_next(self):
        return self.make_iir()

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

    def make_next(self):
        return self.make_module_src()


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

    def make_next(self):
        return self.make_ext_sources()


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

    def make_next(self):
        ctx["ext_module_name"] = ext_mod_name = "_" + ctx["name"]
        return self.make_module_src(
            ctx["ext_module_name"], ctx["output_path"] / ctx["ext_module_name"]
        )

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
        builder = BuildWrapper.from_context(
            ctx,
            frontend=frontend,
            backend=backend,
            build_options=build_options,
            externals=externals or {},
        )
        if not bindings and not backend.BINDINGS_LANGUAGES:
            stage = builder
            while hasattr(stage, "make_next"):
                stage = stage.make_next()
            mod_f = output_path / f'{stage.ctx["name"]}.py'
            mod_f.write_text(stage.module)
        elif compile_bindings:
            package_path = files_from_ctx(builder.ctx)
            shutil.copytree(package_path, output_path / builder.ctx["name"])
        else:
            stage_iir = builder.make_ir().make_iir()
            ext_src = stage_iir.make_ext_sources()
            header_f = output_path / f'{ext_src.ctx["name"]}.hpp'
            impl_f = output_path / f'{ext_src.ctx["name"]}{ext_src.impl_ext}'
            header_f.write_text(ext_src.header)
            impl_f.write_text(ext_src.impl)
            if bindings == "python":
                pybind_f = output_path / f"bindings.cpp"
                pybind_f.write_text(ext_src.bindings)
                ext_mod_name = "_" + ext_src.ctx["name"]
                mod_src = ext_src.make_module_src(
                    ext_module_name=ext_mod_name, ext_module_path=output_path / ext_mod_name
                )
                mod_f = output_path / f'{ext_src.ctx["name"]}.py'
                mod_f.write_text(mod_src.module)
