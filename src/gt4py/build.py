"""
Build process management tools.

Data structures to pass information along and facilitate the entire build process.
Allows fine grained control of build stages.
"""
import abc
import typing
import logging
import pathlib

from cached_property import cached_property

import gt4py
import gt4py.frontend
import gt4py.backend
from gt4py.gtscript_impl import _set_arg_dtypes

LOGGER = logging.Logger("GT4Py build logger")


class BuildContext:
    """
    Primary datastructure for the build process.

    Keeps references to the chosen frontend and backend, contains user chosen options
    as well as default choices and information about current and previous build stages.

    The context is modified by build stages and passed from one build stage to the next.
    """

    MODULE_LOGGER = LOGGER

    def __init__(self, definition, **kwargs):
        """BuildContext can be constructed from a function definition."""
        self.validate_kwargs(kwargs)
        self._data = {}
        self.update({k: v for k, v in kwargs.items() if v is not None})
        self._data["definition"] = definition
        self._set_no_replace("module", getattr(definition, "__module__", ""))
        self._set_no_replace("name", definition.__name__)
        self._set_no_replace("externals", {})
        self._set_no_replace("qualified_name", f"{self._data['module']}.{self._data['name']}")
        self._set_no_replace("build_info", {})
        self._set_no_replace(
            "options",
            gt4py.definitions.BuildOptions(
                name=self._data["name"],
                module=self._data["module"],
                build_info=self._data["build_info"],
            ),
        )
        self.validate()

    @classmethod
    def validate_kwargs(cls, data):
        """
        Validate input kwargs to the constructor.

        Raise ValueError with messages for all invalid kwargs if any are found.
        """
        messages = []
        build_info = data.get("build_info")
        if build_info is not None and not isinstance(build_info, dict):
            messages.append(f"Invalid 'build_info' dictionary ('{build_info}')")
        dtypes = data.get("dtypes")
        if dtypes is not None and not isinstance(dtypes, dict):
            messages.append(f"Invalid 'dtypes' dictionary ('{dtypes}')")
        externals = data.get("externals")
        if externals is not None and not isinstance(externals, dict):
            messages.append(f"Invalid 'externals' dictionary ('{externals}')")
        name = data.get("name")
        if name is not None and not isinstance(name, str):
            messages.append(f"Invalid 'name' string ('{name}')")
        rebuild = data.get("rebuild")
        if rebuild is not None and not isinstance(rebuild, bool):
            messages.append(f"Invalid 'rebuild' bool value ('{rebuild}')")
        if messages:
            messages = "\n".join(messages)
            raise ValueError(f"Invalid arguments for BuildContext:\n {messages}")

    def validate(self):
        self.validate_kwargs(self)

    def _set_no_replace(self, key, value):
        if not key in self._data:
            self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def pop(self, key, default=None):
        return self._data.pop(key, default)

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value: typing.Any):
        if key == "backend":
            value = self._process_backend(value)
        elif key == "frontend":
            value = self._process_frontend(value)
        self._data[key] = value

    def update(self, data: typing.Mapping):
        for key, value in data.items():
            self[key] = value

    def __contains__(self, key: str):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def __eq__(self, other):
        return self._data == other._data

    @classmethod
    def _process_frontend(cls, frontend: typing.Union[str, type]):
        original_value = frontend
        if not isinstance(frontend, type):
            frontend = gt4py.frontend.from_name(frontend)
        if not frontend:
            raise ValueError(f"invalid frontend {original_value}")
        return frontend

    @classmethod
    def _process_backend(cls, backend: typing.Union[str, type]):
        original_value = backend
        if not isinstance(backend, type):
            backend = gt4py.backend.from_name(backend)
        if not backend:
            raise ValueError(f"invalid backend {original_value}")
        return backend


class BuildStage(abc.ABC):
    """Manage transition to the next build stage, adding information to the context."""

    def __init__(self, ctx):
        LOGGER.info("%s created.", self.__class__.__name__)
        self.ctx = ctx

    @abc.abstractmethod
    def _make(self):
        pass

    def make(self):
        self._make()
        LOGGER.info("%s built.", self.__class__.__name__)
        return self

    @abc.abstractmethod
    def next_stage(self):
        pass

    def make_next(self):
        if self.is_final():
            return None
        return self.next_stage()(self.ctx).make()

    def is_final(self):
        return False


class BeginStage(BuildStage):
    """This stage is at the beginning of every build process."""

    def _make(self):
        pass

    def next_stage(self):
        return IRStage


class IRStage(BuildStage):
    """
    Internal Representation stage.

    Make context requirements:

    * `frontend`
    * `backend`
    * `options`

    Make context modifications:

    * `options_id` is generated using the `backend` from `options`
    * `id` is set to the stencil ID generated by the `frontend`
    * `ir` is set to the IR generated by `frontend`
    """

    def _make(self):
        if "ir" in self.ctx:
            return
        frontend = self.ctx["frontend"]
        backend = self.ctx["backend"]
        self.ctx["options_id"] = backend.get_options_id(self.ctx["options"])
        self.ctx["id"] = frontend.get_stencil_id(
            qualified_name=self.ctx["qualified_name"],
            definition=self.ctx["definition"],
            externals=self.ctx["externals"],
            options_id=self.ctx["options_id"],
        )
        self.ctx["ir"] = frontend.generate(
            definition=self.ctx["definition"],
            externals=self.ctx["externals"],
            options=self.ctx["options"],
        )

    def next_stage(self):
        return IIRStage


class IIRStage(BuildStage):
    """
    Internal Implementation Representation stage.

    Make context requirements:

    * `ir`
    * `backend`
    * `options`

    Make context modifications:

    * `iir` is generated through gt4py.analysis.transform
    """

    def _make(self):
        if "iir" in self.ctx:
            return
        backend = self.ctx["backend"]
        backend._check_options(self.ctx["options"])
        self.ctx["iir"] = gt4py.analysis.transform(
            definition_ir=self.ctx["ir"], options=self.ctx["options"]
        )

    def next_stage(self):
        return SourceStage


class SourceStage(BuildStage):
    """
    Implementation language source for the chosen backend.

    Make context requirements:

    * `iir`
    * `id` if backend generates python directly
    * `backend`
    * `options`
    * `bindings` if backend supports bindings and they should be generated

    Make context modifications:

    * `src` is generated via the backend and should be a mapping of filenames to source strings
    """

    def _make(self):
        if "src" in self.ctx:
            return
        backend = self.ctx["backend"]
        if not backend.BINDINGS_LANGUAGES:
            self._make_py_module()
        else:
            self._make_lang_src()

    def _make_lang_src(self):
        backend = self.ctx["backend"]
        generator = backend.PYEXT_GENERATOR_CLASS(
            self.ctx["name"],
            "_" + self.ctx["name"],
            backend._CPU_ARCHITECTURE,
            self.ctx["options"],
        )
        self.ctx["src"] = generator(self.ctx["iir"])
        if "computation.src" in self.ctx["src"]:
            self.ctx["src"][f"computation.{backend.SRC_EXTENSION}"] = self.ctx["src"].pop(
                "computation.src"
            )
        if "bindings.cpp" in self.ctx["src"]:
            self.ctx["bindings_src"] = self.ctx.get("bindings_src", {})
            self.ctx["bindings_src"]["bindings.cpp"] = self.ctx["src"].pop("bindings.cpp")

    def _make_py_module(self):
        backend = self.ctx["backend"]
        generator_options = self.ctx["options"].as_dict()
        generator = backend.GENERATOR_CLASS(backend, options=generator_options)
        self.ctx["generator_options"] = generator_options
        self.ctx["src"] = {
            f"{self.ctx['name']}.py": generator(
                self.ctx["id"], self.ctx["iir"], override_stencil_class_name=self.ctx["name"]
            )
        }

    def is_final(self):
        if self.ctx["backend"].BINDINGS_LANGUAGES and self.ctx.get("bindings", None):
            return False
        return True

    def next_stage(self):
        return BindingsStage


class BindingsStage(BuildStage):
    """
    Language bindings stage.

    Make context requirements:

    * `backend` must support language bindings
    * `bindings` must be a non-empty list of languages supported by `backend`
    * `options`
    * `iir`
    * `id`
    * `pyext_file_path_final`
    * `bindings_src`, optional, may contain a key `bindings.cpp`.

    Make context modifications:

    * `bindings_src` is generated via the backend and should be a mapping of filenames to source strings
    * `pyext_module_name` is taken from `pyext_file_path_final`
    * `pyext_generator_options` records the options passed to the python extension generator.
    """

    def is_final(self):
        return not self.ctx.get("compile_bindings", False)

    def _make(self):
        if "python" in self.ctx["bindings"]:
            backend = self.ctx["backend"]
            pyext_module_name = self.ctx["pyext_file_path_final"].split(".")[0]
            generator_options = self.ctx["options"].as_dict()
            generator_options["pyext_module_name"] = pyext_module_name
            generator_options["pyext_file_path"] = self.ctx["pyext_file_path_final"]
            self.ctx["pyext_generator_options"] = generator_options
            self.ctx["pyext_module_name"] = generator_options["pyext_module_name"]
            generator = backend.GENERATOR_CLASS(backend, options=generator_options)
            bindings_src = {}
            bindings_src["python"] = {
                "bindings.cpp": self.ctx.get("bindings_src", {}).get("bindings.cpp", ""),
                f"{self.ctx['name']}.py": generator(
                    self.ctx["id"], self.ctx["iir"], override_stencil_class_name=self.ctx["name"]
                ),
            }
            self.ctx["bindings_src"] = bindings_src

    def next_stage(self):
        if not self.is_final():
            return CompileBindingsStage
        return None


class CompileBindingsStage(BuildStage):
    """
    Compile language bindings stage.

    Make context requirements:

    * `backend` must support language bindings
    * `bindings` must be a non-empty list of languages supported by `backend`
    * `compile_bindings` must be True
    * `bindings_src_files` must be a dict with a key for each language in `bindings` and a list for each
        key containing existing source files for the compilation
    * `pyext_module_name` must be the name of the target module
    * `pyext_module_path` must be an existing directory
    * `pyext_opts`, optional options for the pyext builder

    Side effects:

    * `pyext_module_path`/`pyext_module_name`.so will be a compiled python extension object file
    """

    def is_final(self):
        return True

    def _make(self):
        if "python" in self.ctx["bindings"] and self.ctx["compile_bindings"]:
            backend = self.ctx["backend"]
            pyext_opts = self.ctx.get("pyext_opts", {})
            module_name, file_path = backend.build_pyext(
                self.ctx["pyext_module_name"],
                sources=self.ctx["bindings_src_files"]["python"],
                build_path=self.ctx["pyext_module_path"],
                target_path=self.ctx["pyext_module_path"],
                **pyext_opts,
            )
            tmp_pyext_file = pathlib.Path(file_path)
            pyext_file = pathlib.Path(self.ctx["pyext_file_path_final"])
            tmp_pyext_file.rename(pyext_file)
            self.ctx["pyext_file_path"] = pyext_file

    def next_stage(self):
        return None


class LazyStencil:
    """
    A stencil object which defers compilation until it is needed.

    Usually obtained using the :func:`gt4py.gtscript.lazy_stencil` decorator, not directly instanciated.

    This is done by keeping all the necessary information in a :class:`gt4py.build.BuildContex`
    object in the `ctx` attribute.

    Compilation happens implicitly on first access to the `implementation` property.
    A step by step compilation process can be initiated by calling :func:`gt4py.build.LazyStencil.begin_build`.
    """

    def __init__(self, context):
        self.ctx = context

    @cached_property
    def implementation(self):
        """
        The compiled backend-specific python callable which executes the stencil.

        Compilation happens at first access, the result is cached and should consecutively be
        accessible without overhead (not rigorously tested / benchmarked).
        """
        keys = ["backend", "definition", "build_info", "dtypes", "externals", "name", "rebuild"]
        kwargs = {k: v for k, v in self.ctx._data.items() if k in keys}
        kwargs.update(self.ctx["options"].as_dict()["backend_opts"])
        impl = self._jit_build()
        return impl

    def _jit_build(self):
        """Build and load."""
        from gt4py import loader as gt_loader

        _set_arg_dtypes(self.ctx["definition"], self.ctx.get("dtypes"))
        return gt_loader.gtscript_loader(
            self.ctx["definition"],
            backend=self.ctx["backend"].name,
            build_options=self.ctx["options"],
            externals=self.ctx["externals"],
        )

    @property
    def backend(self):
        """
        The backend to be used for compilation.

        Does not trigger a build.
        """
        return self.ctx["backend"]

    @property
    def field_info(self):
        """
        Access the compiled stencil object's `field_info` attribute.

        Triggers a build if necessary.
        """
        return self.implementation.field_info

    def begin_build(self):
        """
        Create a :class:`gt4py.build.BeginStage` from the build context.

        No generation / compilation is done yet, but the process can be stepped through
        by calling :func:`gt4py.build.BeginStage.make_next` on the result.
        """
        return BeginStage(self.ctx).make()

    def check_syntax(self):
        """
        Create the gtscript IR for the stencil, failing on syntax errors.

        This step is cached and will be skipped on subsequent builds starting from
        :func:`gt4py.gtscript.LazyStencil.begin_build`.
        """
        BeginStage(self.ctx).make().make_next()

    def __call__(self, *args, **kwargs):
        """
        Execute the stencil, building the stencil if necessary.
        """
        self.implementation(*args, **kwargs)
