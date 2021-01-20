# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import abc
import copy
import hashlib
import numbers
import os
import pathlib
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union

import jinja2

import gt4py
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.storage.default_parameters import StorageDefaults

from . import pyext_builder


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder
    from gt4py.stencil_object import StencilObject

REGISTRY = gt_utils.Registry()


def from_name(name: str) -> Type["Backend"]:
    return REGISTRY.get(name, None)


def register(backend_cls):
    assert issubclass(backend_cls, Backend) and backend_cls.name is not None

    if not isinstance(backend_cls.name, str):
        raise ValueError(
            "Invalid 'name' attribute ('{name}') in backend class '{cls}'".format(
                name=backend_cls.name, cls=backend_cls
            )
        )

    if isinstance(backend_cls.storage_defaults, gt4py.storage.default_parameters.StorageDefaults):
        gt4py.storage.register_storage_defaults(
            name=backend_cls.name, defaults=backend_cls.storage_defaults
        )
    else:
        raise ValueError(f"Invalid 'storage_defaults' attribute in backend class '{backend_cls}'")

    return REGISTRY.register(backend_cls.name, backend_cls)


def remove(backend_key):
    REGISTRY.pop(backend_key)
    gt4py.storage.default_parameters.REGISTRY.pop(backend_key)


class Backend(abc.ABC):

    #: Backend name
    name: ClassVar[str]

    #: Backend-specific options:
    #:
    #:  + info:
    #:    - versioning: is versioning on?
    #:    - description [optional]
    #:    - type
    options: ClassVar[Dict[str, Any]]

    #: the processing unit where the generated code will run, one of , "cpu" or "gpu"
    compute_device: ClassVar[str] = "cpu"

    #: Backend-specific default storage parametrization
    storage_defaults: ClassVar[StorageDefaults] = StorageDefaults()

    #: Language support:
    #:
    #:  - "computation": name of the language in which the computation is implemented.
    #:  - "bindings": names of supported language bindings / wrappers.
    #:    If a high-level language is compatible out-of-the-box, it should not be listed.
    #:
    #:  Languages should be spelled using the official spelling
    #:  but lower case ("python", "fortran", "rust").
    languages: ClassVar[Optional[Dict[str, Any]]] = None

    # __impl_opts:
    #   "disable-code-generation": bool
    #   "disable-cache-validation": bool

    builder: "StencilBuilder"

    def __init__(self, builder: "StencilBuilder"):
        self.builder = builder

    @classmethod
    def filter_options_for_id(
        cls, options: gt_definitions.BuildOptions
    ) -> gt_definitions.BuildOptions:
        """Filter a copy for options that should not be part of the stencil ID."""
        filtered_options = copy.deepcopy(options)
        id_names = set(name for name, info in cls.options.items() if info["versioning"])
        non_id_names = set(filtered_options.backend_opts.keys()) - id_names
        for name in non_id_names:
            del filtered_options.backend_opts[name]

        return filtered_options

    @abc.abstractmethod
    def load(self) -> Optional[Type["StencilObject"]]:
        """
        Load the stencil class from the generated python module.

        type:
            The generated stencil class after loading through python's import API


        This assumes that :py:meth:`Backend.generate` has been called already on the same stencil.
        The python module therefore already exists and any python extensions have been compiled.
        In case the stencil changed since the last generate call, it will not be rebuilt.
        """
        pass

    @abc.abstractmethod
    def generate(self) -> Type["StencilObject"]:
        """
        Generate the stencil class from GTScript's internal representation.

        Returns
        -------
        type:
            The generated stencil class after loading through python's import API

        In case the stencil class for the given ID is found in cache, this method can avoid
        rebuilding it. Rebuilding can however be triggered through the :code:`options` parameter
        if supported by the backend.
        """
        pass

    @property
    def extra_cache_info(self) -> Dict[str, Any]:
        """Provide additional data to be stored in cache info file (sublass hook)."""
        return {}

    @property
    def extra_cache_validation_keys(self) -> List[str]:
        """List keys from extra_cache_info to be validated during consistency check."""
        return []


class CLIBackendMixin(Backend):
    @abc.abstractmethod
    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        """
        Generate the computation source code in a way agnostic of the way it is going to be used.

        Returns
        -------
        Dict[str, str | Dict] of source file names / directories -> contents:
            If a key's value is a string it is interpreted as a file name and the value as the
            source code of that file
            If a key's value is a Dict, it is interpreted as a directory name and it's
            value as a nested file hierarchy to which the same rules are applied recursively.
            The root path is relative to the build directory.

        Raises
        ------
        NotImplementedError
            If the backend does not support usage outside of JIT compilation / generation.

        Example
        -------
        .. code-block:: python

            def mystencil(...):
                ...

            options = BuildOptions(name="mystencil", ...)
            ir = frontend.generate(mystencil, {}, options)
            stencil_src = backend.generate_computation(ir, options)

            print(stencil_src)

            # this might be output from a fictional backend:
            {
                "mystencil_project": {
                    "src": {
                        "stencil.cpp",
                        "helpers.cpp"
                    },
                    "include": {
                        "stencil.hpp"
                    },
                }
            }

        This can now be automatically be turned into a folder hierarchy that makes sense
        and can be incorporated into an external build system.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        """
        Generate bindings source code from ``language_name`` to the target language of the backend.

        Returns
        -------
        Analog to :py:meth:`generate_computation` but containing bindings source code, The
        dictionary contains a tree of directories with leaves being a mapping from filename to
        source code pairs, relative to the build directory.

        Raises
        ------
        RuntimeError
            If the backend does not support the bindings language

        """
        languages = getattr(self, "languages", {"bindings": {}})
        name = getattr(self, "name", "")
        if language_name not in languages["bindings"]:
            raise NotImplementedError(
                f"Backend {name} does not implement bindings for {language_name}"
            )
        return {}


class BaseBackend(Backend):

    MODULE_GENERATOR_CLASS: ClassVar[Type["BaseModuleGenerator"]]

    def load(self) -> Optional[Type["StencilObject"]]:
        stencil_class = None
        if self.builder.stencil_id is not None:
            self.check_options(self.builder.options)
            validate_hash = not self.builder.options._impl_opts.get(
                "disable-cache-validation", False
            )
            if self.builder.caching.is_cache_info_available_and_consistent(
                validate_hash=validate_hash
            ):
                stencil_class = self._load()

        return stencil_class

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)
        return self.make_module()

    def _load(self) -> Type["StencilObject"]:
        stencil_class_name = self.builder.class_name
        file_name = str(self.builder.module_path)
        stencil_module = gt_utils.make_module_from_file(stencil_class_name, file_name)
        stencil_class = getattr(stencil_module, stencil_class_name)
        stencil_class.__module__ = self.builder.module_qualname
        stencil_class._gt_id_ = self.builder.stencil_id.version
        stencil_class._file_name = file_name
        stencil_class.definition_func = staticmethod(self.builder.definition)

        return stencil_class

    def check_options(self, options: gt_definitions.BuildOptions) -> None:
        assert self.options is not None
        unknown_options = set(options.backend_opts.keys()) - set(self.options.keys())
        if unknown_options:
            raise ValueError("Unknown backend options: '{}'".format(unknown_options))

    def make_module(
        self,
        **kwargs: Any,
    ) -> Type["StencilObject"]:
        file_path = self.builder.module_path
        module_source = self.make_module_source(**kwargs)

        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(module_source)
            self.builder.caching.update_cache_info()

        return self._load()

    def make_module_source(
        self, *, args_data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        """Generate the module source code with or without stencil id."""
        args_data = args_data or self.make_args_data_from_iir(self.builder.implementation_ir)
        source = self.MODULE_GENERATOR_CLASS()(args_data, self.builder, **kwargs)
        return source

    @staticmethod
    def make_args_data_from_iir(implementation_ir: gt_ir.StencilImplementation) -> Dict[str, Any]:
        data: Dict[str, Any] = {"field_info": {}, "parameter_info": {}, "unreferenced": {}}

        # Collect access type per field
        out_fields = set()
        for ms in implementation_ir.multi_stages:
            for sg in ms.groups:
                for st in sg.stages:
                    for acc in st.accessors:
                        if (
                            isinstance(acc, gt_ir.FieldAccessor)
                            and acc.intent == gt_ir.AccessIntent.READ_WRITE
                        ):
                            out_fields.add(acc.symbol)

        for arg in implementation_ir.api_signature:
            if arg.name in implementation_ir.fields:
                access = (
                    gt_definitions.AccessKind.READ_WRITE
                    if arg.name in out_fields
                    else gt_definitions.AccessKind.READ_ONLY
                )
                if arg.name not in implementation_ir.unreferenced:
                    data["field_info"][arg.name] = gt_definitions.FieldInfo(
                        access=access,
                        dtype=implementation_ir.fields[arg.name].data_type.dtype,
                        boundary=implementation_ir.fields_extents[arg.name].to_boundary(),
                    )
                else:
                    data["field_info"][arg.name] = None
            else:
                if arg.name not in implementation_ir.unreferenced:
                    data["parameter_info"][arg.name] = gt_definitions.ParameterInfo(
                        dtype=implementation_ir.parameters[arg.name].data_type.dtype
                    )
                else:
                    data["parameter_info"][arg.name] = None

        data["unreferenced"] = implementation_ir.unreferenced

        return data


class PurePythonBackendCLIMixin(CLIBackendMixin):
    """Mixin for CLI support for backends deriving from BaseBackend."""

    builder: "StencilBuilder"

    #: stencil python source generator method:
    #:  In order to use this mixin, the backend class must implement
    #:  a :py:meth:`make_module_source` method or derive from
    #:  :py:meth:`BaseBackend`.
    make_module_source: Callable

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        file_name = self.builder.module_path.name
        source = self.make_module_source(implementation_ir=self.builder.implementation_ir)
        return {str(file_name): source}

    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        """Pure python backends typically will not support bindings."""
        return super().generate_bindings(language_name)


class BasePyExtBackend(BaseBackend):
    @property
    def pyext_module_name(self) -> str:
        return self.builder.module_name + "_pyext"

    @property
    def pyext_module_path(self) -> str:
        return self.builder.module_qualname + "_pyext"

    @property
    def pyext_class_name(self) -> str:
        return self.builder.class_name + "_pyext"

    @property
    def pyext_build_dir_path(self) -> pathlib.Path:
        return self.builder.pkg_path.joinpath(self.pyext_module_name + "_BUILD")

    @property
    def extra_cache_info(self) -> Dict[str, Any]:
        pyext_file_path = self.builder.backend_data.get("pyext_file_path", None)
        pyext_md5 = ""
        if pyext_file_path:
            pyext_md5 = hashlib.md5(pathlib.Path(pyext_file_path).read_bytes()).hexdigest()
        return {
            **super().extra_cache_info,
            "pyext_file_path": pyext_file_path,
            "pyext_md5": pyext_md5,
        }

    @property
    def extra_cache_validation_keys(self) -> List[str]:
        keys = super().extra_cache_validation_keys
        if self.extra_cache_info["pyext_md5"]:
            keys.append("pyext_md5")
        return keys

    @abc.abstractmethod
    def generate(self) -> Type["StencilObject"]:
        pass

    def build_extension_module(
        self,
        pyext_sources: Dict[str, Any],
        pyext_build_opts: Dict[str, str],
        *,
        uses_cuda: bool = False,
    ) -> Tuple[str, str]:
        # Build extension module
        pyext_build_path = pathlib.Path(
            os.path.relpath(self.pyext_build_dir_path, pathlib.Path.cwd())
        )
        pyext_build_path.mkdir(parents=True, exist_ok=True)
        sources = []
        for key, source in pyext_sources.items():
            src_file_path = pyext_build_path / key
            src_ext = src_file_path.suffix
            if src_ext not in [".h", ".hpp"]:
                sources.append(str(src_file_path))

            if source is not gt_utils.NOTHING:
                src_file_path.write_text(source)

        pyext_target_file_path = self.builder.pkg_path
        qualified_pyext_name = self.pyext_module_path

        pyext_build_args: Dict[str, Any] = dict(
            name=qualified_pyext_name,
            sources=sources,
            build_path=str(pyext_build_path),
            target_path=str(pyext_target_file_path),
            **pyext_build_opts,
        )

        if uses_cuda:
            module_name, file_path = pyext_builder.build_pybind_cuda_ext(**pyext_build_args)
        else:
            module_name, file_path = pyext_builder.build_pybind_ext(**pyext_build_args)

        assert module_name == qualified_pyext_name

        self.builder.with_backend_data(
            {"pyext_module_name": module_name, "pyext_file_path": file_path}
        )

        return module_name, file_path


class BaseModuleGenerator(abc.ABC):

    SOURCE_LINE_LENGTH = 120
    TEMPLATE_INDENT_SIZE = 4
    DOMAIN_ARG_NAME = "_domain_"
    ORIGIN_ARG_NAME = "_origin_"
    SPLITTERS_NAME = "_splitters_"

    TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "stencil_module.py.in")

    _builder: Optional["StencilBuilder"]
    args_data: Dict[str, Any]
    template: jinja2.Template

    def __init__(self, builder: Optional["StencilBuilder"] = None):
        self._builder = builder
        self.args_data = {}
        with open(self.TEMPLATE_PATH, "r") as f:
            self.template = jinja2.Template(f.read())

    def __call__(
        self,
        args_data: Dict[str, Any],
        builder: Optional["StencilBuilder"] = None,
        **kwargs: Any,
    ) -> str:
        """Generate source code for a Python module containing a StencilObject."""
        if builder:
            self._builder = builder
        self.args_data = args_data

        definition_ir = self.builder.definition_ir

        if definition_ir.sources is not None:
            sources = {
                key: gt_utils.text.format_source(value, line_length=self.SOURCE_LINE_LENGTH)
                for key, value in definition_ir.sources
            }
        else:
            sources = {}

        if definition_ir.externals:
            constants = {
                name: repr(value)
                for name, value in definition_ir.externals.items()
                if isinstance(value, numbers.Number)
            }
        else:
            constants = {}

        options = {
            key: value
            for key, value in self.builder.options.as_dict().items()
            if key not in ["build_info"]
        }

        parallel_axes = definition_ir.domain.parallel_axes or []
        sequential_axis = definition_ir.domain.sequential_axis.name
        domain_info = repr(
            gt_definitions.DomainInfo(
                parallel_axes=tuple(ax.name for ax in parallel_axes),
                sequential_axis=sequential_axis,
                ndims=len(parallel_axes) + (1 if sequential_axis else 0),
            )
        )

        module_source = self.template.render(
            imports=self.generate_imports(),
            module_members=self.generate_module_members(),
            class_name=self.builder.class_name,
            class_members=self.generate_class_members(),
            docstring=definition_ir.docstring,
            gt_backend=self.backend_name,
            gt_source=sources,
            gt_domain_info=domain_info,
            gt_field_info=repr(self.args_data["field_info"]),
            gt_parameter_info=repr(self.args_data["parameter_info"]),
            gt_constants=constants,
            gt_options=options,
            stencil_signature=self.generate_signature(),
            field_names=self.args_data["field_info"].keys(),
            param_names=self.args_data["parameter_info"].keys(),
            pre_run=self.generate_pre_run(),
            post_run=self.generate_post_run(),
            implementation=self.generate_implementation(),
        )
        if options["format_source"]:
            module_source = gt_utils.text.format_source(
                module_source, line_length=self.SOURCE_LINE_LENGTH
            )

        return module_source

    @property
    def builder(self) -> "StencilBuilder":
        """
        Expose the builder reference.

        Raises a runtime error if the builder reference is not initialized.
        This is necessary because other parts of the public API depend on it before it is
        guaranteed to be initialized.
        """
        if not self._builder:
            raise RuntimeError("Builder attribute not initialized!")
        return self._builder

    @property
    def backend_name(self) -> str:
        return self.builder.backend.name

    @abc.abstractmethod
    def generate_implementation(self) -> str:
        pass

    def generate_imports(self) -> str:
        source = ""
        return source

    def generate_module_members(self) -> str:
        source = ""
        return source

    def generate_class_members(self) -> str:
        source = ""
        return source

    def generate_signature(self) -> str:
        args = []
        keyword_args = ["*"]
        for arg in self.builder.definition_ir.api_signature:
            if arg.is_keyword:
                if arg.default is not gt_ir.Empty:
                    keyword_args.append(
                        "{name}={default}".format(name=arg.name, default=arg.default)
                    )
                else:
                    keyword_args.append(arg.name)
            else:
                if arg.default is not gt_ir.Empty:
                    args.append("{name}={default}".format(name=arg.name, default=arg.default))
                else:
                    args.append(arg.name)

        if len(keyword_args) > 1:
            args.extend(keyword_args)
        signature = ", ".join(args)

        return signature

    def generate_pre_run(self) -> str:
        source = ""
        return source

    def generate_post_run(self) -> str:
        source = ""
        return source


class PyExtModuleGenerator(BaseModuleGenerator):

    pyext_module_name: str
    pyext_file_path: str

    def __init__(self):
        super().__init__()
        self.pyext_module_name = ""
        self.pyext_file_path = ""

    def __call__(
        self,
        args_data: Dict[str, Any],
        builder: Optional["StencilBuilder"] = None,
        **kwargs: Any,
    ) -> str:
        self.pyext_module_name = kwargs["pyext_module_name"]
        self.pyext_file_path = kwargs["pyext_file_path"]
        return super().__call__(args_data, builder, **kwargs)

    def generate_imports(self) -> str:
        source = """
from gt4py import utils as gt_utils
        """
        if self.builder.implementation_ir.multi_stages:
            source += """
pyext_module = gt_utils.make_module_from_file(
        "{pyext_module_name}", "{pyext_file_path}", public_import=True
    )
        """.format(
                pyext_module_name=self.pyext_module_name, pyext_file_path=self.pyext_file_path
            )
        return source

    def generate_implementation(self) -> str:
        definition_ir = self.builder.definition_ir
        sources = gt_utils.text.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        api_fields = set(field.name for field in definition_ir.api_fields)
        for arg in definition_ir.api_signature:
            if arg.name not in self.args_data["unreferenced"]:
                if from_name(self.backend_name).storage_defaults.device == "cpu":
                    args.append(f"np.asarray({arg.name})")
                else:
                    args.append(arg.name)
                if arg.name in api_fields:
                    args.append("list(_origin_['{}'])".format(arg.name))

        # only generate implementation if any multi_stages are present. e.g. if no statement in the
        # stencil has any effect on the API fields, this may not be the case since they could be
        # pruned.
        if self.builder.implementation_ir.has_effect:
            source = """
# Load or generate a GTComputation object for the current domain size
pyext_module.run_computation(list(_domain_), {run_args}, exec_info)
""".format(
                run_args=", ".join(args)
            )
            sources.extend(source.splitlines())
        else:
            sources.extend("\n")

        return sources.text


class CUDAPyExtModuleGenerator(PyExtModuleGenerator):
    def generate_implementation(self) -> str:
        source = (
            super().generate_implementation()
            + """
cupy.cuda.Device(0).synchronize()
    """
        )
        return source

    def generate_imports(self) -> str:
        source = (
            """
import cupy
"""
            + super().generate_imports()
        )
        return source
