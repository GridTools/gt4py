# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Optional, Tuple, Type, Union

import jinja2

from gt4py import analysis as gt_analysis
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.stencil_object import StencilObject

from . import pyext_builder


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder

REGISTRY = gt_utils.Registry()


def from_name(name: str) -> Type:
    return REGISTRY.get(name, None)


def register(backend_cls: Type["Backend"]) -> None:
    assert issubclass(backend_cls, Backend) and backend_cls.name is not None

    if isinstance(backend_cls.name, str):
        return REGISTRY.register(backend_cls.name, backend_cls)

    else:
        raise ValueError(
            "Invalid 'name' attribute ('{name}') in backend class '{cls}'".format(
                name=backend_cls.name, cls=backend_cls
            )
        )


class Backend(abc.ABC):

    #: Backend name
    name: ClassVar[str]

    #: Backend-specific options:
    #:
    #:  + info:
    #:    - versioning: is versioning on?
    #:    - description [optional]
    #:
    options: ClassVar[Dict[str, Any]]

    #: Backend-specific storage parametrization:
    #:
    #:  - "alignment": in bytes
    #:  - "device": "cpu" | "gpu"
    #:  - "layout_map": callback converting a mask to a layout
    #:  - "is_compatible_layout": callback checking if a storage has compatible layout
    #:  - "is_compatible_type": callback checking if storage has compatible type
    storage_info: ClassVar[Dict[str, Any]]

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
    def load(self) -> Optional[Type[StencilObject]]:
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
    def generate(self) -> Type[StencilObject]:
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
        return {}

    @property
    def extra_cache_validation_data(self) -> Dict[str, Any]:
        return {}


class CLIBackendMixin:
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


class BaseBackend(Backend):

    MODULE_GENERATOR_CLASS: ClassVar[Type["BaseModuleGenerator"]]

    def _check_options(self, options: gt_definitions.BuildOptions) -> None:
        assert self.options is not None
        unknown_options = set(options.backend_opts.keys()) - set(self.options.keys())
        if unknown_options:
            raise ValueError("Unknown backend options: '{}'".format(unknown_options))

    def _load(self) -> Type[StencilObject]:
        stencil_class_name = self.builder.class_name
        file_name = str(self.builder.module_path)
        stencil_module = gt_utils.make_module_from_file(stencil_class_name, file_name)
        stencil_class = getattr(stencil_module, stencil_class_name)
        stencil_class.__module__ = self.builder.module_qualified_name
        stencil_class._gt_id_ = self.builder.stencil_id.version
        stencil_class.definition_func = staticmethod(self.builder.definition)

        return stencil_class

    def load(self) -> Optional[Type[StencilObject]]:
        stencil_class = None
        if self.builder.stencil_id is not None:
            self._check_options(self.builder.options)
            validate_hash = not self.builder.options._impl_opts.get(
                "disable-cache-validation", False
            )
            if self.builder.caching.is_cache_info_available_and_consistent(
                validate_hash=validate_hash
            ):
                stencil_class = self._load()

        return stencil_class

    def _generate_module(
        self, *, extra_cache_info: Optional[Dict[str, Any]] = None, **kwargs: Any,
    ) -> Type[StencilObject]:
        file_path = self.builder.module_path
        module_source = self._generate_module_source(**kwargs)

        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(module_source)
            extra_cache_info = {**self.extra_cache_info, **(extra_cache_info or {})}
            self.builder.caching.update_cache_info()

        return self._load()

    def _generate_module_source(self, **kwargs: Any) -> str:
        """Generate the module source code with or without stencil id."""
        source = self.MODULE_GENERATOR_CLASS(builder=self.builder)(**kwargs)
        return source

    def generate(self) -> Type[StencilObject]:
        self._check_options(self.builder.options)
        return self._generate_module()


class PurePythonBackendCLIMixin(CLIBackendMixin):
    """Mixin for CLI support for backends deriving from BaseBackend."""

    builder: "StencilBuilder"

    #: stencil python source generator method:
    #:  In order to use this mixin, the backend class must implement
    #:  a :py:meth:`_generate_module_source` method or derive from
    #:  :py:meth:`BaseBackend`.
    _generate_module_source: Callable

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        file_name = self.builder.module_path.name
        source = self._generate_module_source(implementation_ir=self.builder.implementation_ir)
        return {str(file_name): source}


class BasePyExtBackend(BaseBackend):
    @property
    def pyext_module_name(self) -> str:
        return self.builder.module_name + "_pyext"

    @property
    def pyext_module_path(self) -> str:
        return self.builder.module_qualified_name + "_pyext"

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
    def extra_cache_validation_data(self) -> Dict[str, Any]:
        if "pyext_file_path" not in self.builder.backend_data:
            return {}
        pyext_file_path = pathlib.Path(self.builder.backend_data["pyext_file_path"])
        if not pyext_file_path.exists():
            return {}
        pyext_md5 = hashlib.md5(pyext_file_path.read_bytes()).hexdigest()
        return {**super().extra_cache_info, "pyext_md5": pyext_md5}

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

        if uses_cuda:
            module_name, file_path = pyext_builder.build_pybind_cuda_ext(
                qualified_pyext_name,
                sources=sources,
                build_path=str(pyext_build_path),
                target_path=str(pyext_target_file_path),
                **pyext_build_opts,
            )
        else:
            module_name, file_path = pyext_builder.build_pybind_ext(
                qualified_pyext_name,
                sources=sources,
                build_path=str(pyext_build_path),
                target_path=str(pyext_target_file_path),
                **pyext_build_opts,
            )
        assert module_name == qualified_pyext_name

        self.builder.with_backend_data(
            {"pyext_module_name": module_name, "pyext_file_path": file_path}
        )

        return module_name, file_path

    @abc.abstractmethod
    def generate(self) -> Type[StencilObject]:
        pass


class BaseModuleGenerator(abc.ABC):

    SOURCE_LINE_LENGTH = 120
    TEMPLATE_INDENT_SIZE = 4
    DOMAIN_ARG_NAME = "_domain_"
    ORIGIN_ARG_NAME = "_origin_"
    SPLITTERS_NAME = "_splitters_"

    TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "stencil_module.py.in")

    def __init__(self, builder: "StencilBuilder"):
        assert isinstance(builder.backend, BaseBackend)
        self.backend_class = builder.backend
        self.builder = builder
        self.options: gt_definitions.BuildOptions
        self.stencil_id: gt_definitions.StencilID
        self.definition_ir: gt_ir.StencilDefinition
        with open(self.TEMPLATE_PATH, "r") as f:
            self.template = jinja2.Template(f.read())

        self.implementation_ir: gt_ir.StencilImplementation
        self.module_info: Dict[str, Any]

    def __call__(self, module_info: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:

        self.options = self.builder.options
        self.stencil_id = self.builder.stencil_id
        self.definition_ir = self.builder.definition_ir
        self.implementation_ir = kwargs.get("implementation_ir", None)

        if module_info is None:
            # If a `module_info dict` is not explicitly provided by a subclass, it will be
            # generated from a `implementation_ir` object. If this is also not provided, the
            # internal analysis pipeline would be used as a fallback to generate both
            if self.implementation_ir is None:
                self.implementation_ir = self._generate_implementation_ir()
            self.module_info = self._generate_module_info()
        else:
            self.module_info = module_info

        module_info = self.module_info
        stencil_signature = self.generate_signature()
        imports = self.generate_imports()
        module_members = self.generate_module_members()
        class_members = self.generate_class_members()
        implementation = self.generate_implementation()
        pre_run = self.generate_pre_run()
        post_run = self.generate_post_run()

        module_source = self.template.render(
            imports=imports,
            module_members=module_members,
            class_name=self.builder.class_name,
            class_members=class_members,
            docstring=module_info["docstring"],
            gt_backend=self.backend_name,
            gt_source=module_info["sources"],
            gt_domain_info=module_info["domain_info"],
            gt_field_info=repr(module_info["field_info"]),
            gt_parameter_info=repr(module_info["parameter_info"]),
            gt_constants=module_info["gt_constants"],
            gt_options=module_info["gt_options"],
            stencil_signature=stencil_signature,
            field_names=module_info["field_info"].keys(),
            param_names=module_info["parameter_info"].keys(),
            pre_run=pre_run,
            post_run=post_run,
            implementation=implementation,
        )
        module_source = gt_utils.text.format_source(
            module_source, line_length=self.SOURCE_LINE_LENGTH
        )

        return module_source

    @property
    def backend_name(self) -> str:
        return self.backend_class.name

    def _generate_implementation_ir(self) -> gt_ir.StencilImplementation:
        implementation_ir = gt_analysis.transform(self.definition_ir, self.options)
        return implementation_ir

    def _generate_module_info(
        self, field_info: Optional[Dict[str, Optional[gt_definitions.FieldInfo]]] = None
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        implementation_ir = self.implementation_ir

        if self.definition_ir.sources is not None:
            info["sources"].update(
                {
                    key: gt_utils.text.format_source(value, line_length=self.SOURCE_LINE_LENGTH)
                    for key, value in self.definition_ir.sources
                }
            )
        else:
            info["sources"] = {}

        info["docstring"] = implementation_ir.docstring

        parallel_axes = implementation_ir.domain.parallel_axes or []
        sequential_axis = implementation_ir.domain.sequential_axis.name
        info["domain_info"] = repr(
            gt_definitions.DomainInfo(
                parallel_axes=tuple(ax.name for ax in parallel_axes),
                sequential_axis=sequential_axis,
                ndims=len(parallel_axes) + (1 if sequential_axis else 0),
            )
        )

        field_info = field_info or {}
        info["field_info"] = field_info
        parameter_info: Dict[str, Optional[gt_definitions.ParameterInfo]] = {}
        info["parameter_info"] = parameter_info

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
                    field_info[arg.name] = gt_definitions.FieldInfo(
                        access=access,
                        dtype=implementation_ir.fields[arg.name].data_type.dtype,
                        boundary=implementation_ir.fields_extents[arg.name].to_boundary(),
                    )
                else:
                    field_info[arg.name] = None
            else:
                if arg.name not in implementation_ir.unreferenced:
                    parameter_info[arg.name] = gt_definitions.ParameterInfo(
                        dtype=implementation_ir.parameters[arg.name].data_type.dtype
                    )
                else:
                    parameter_info[arg.name] = None

        if implementation_ir.externals:
            info["gt_constants"] = {
                name: repr(value)
                for name, value in implementation_ir.externals.items()
                if isinstance(value, numbers.Number)
            }
        else:
            info["gt_constants"] = {}

        info["gt_options"] = {
            key: value
            for key, value in self.options.as_dict().items()
            if key not in ["build_info"]
        }

        info["unreferenced"] = self.implementation_ir.unreferenced

        return info

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
        for arg in self.definition_ir.api_signature:
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

    @abc.abstractmethod
    def generate_implementation(self) -> str:
        pass


class PyExtModuleGenerator(BaseModuleGenerator):
    def __init__(self, builder: "StencilBuilder"):
        super().__init__(builder)
        self.pyext_module_name = None
        self.pyext_file_path = None

    def __call__(self, module_info: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:

        self.pyext_module_name = kwargs["pyext_module_name"]
        self.pyext_file_path = kwargs["pyext_file_path"]
        return super().__call__(module_info, **kwargs)

    def generate_imports(self) -> str:
        source = """
from gt4py import utils as gt_utils
        """
        if self.implementation_ir.multi_stages:
            source += """
pyext_module = gt_utils.make_module_from_file(
        "{pyext_module_name}", "{pyext_file_path}", public_import=True
    )
        """.format(
                pyext_module_name=self.pyext_module_name, pyext_file_path=self.pyext_file_path
            )
        return source

    def generate_implementation(self) -> str:
        sources = gt_utils.text.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        api_fields = set(field.name for field in self.definition_ir.api_fields)
        for arg in self.definition_ir.api_signature:
            if arg.name not in self.module_info["unreferenced"]:
                args.append(arg.name)
                if arg.name in api_fields:
                    args.append("list(_origin_['{}'])".format(arg.name))

        # only generate implementation if any multi_stages are present. e.g. if no statement in the
        # stencil has any effect on the API fields, this may not be the case since they could be
        # pruned.
        if self.implementation_ir.multi_stages:
            source = """
# Load or generate a GTComputation object for the current domain size
pyext_module.run_computation(list(_domain_), {run_args}, exec_info)
""".format(
                run_args=", ".join(args)
            )
            sources.extend(source.splitlines())
        else:
            source = "\n"

        return sources.text


class CUDAPyExtModuleGenerator(PyExtModuleGenerator):
    def generate_imports(self) -> str:
        source = (
            """
import cupy
"""
            + super().generate_imports()
        )
        return source

    def generate_implementation(self) -> str:
        source = (
            super().generate_implementation()
            + """
cupy.cuda.Device(0).synchronize()
"""
        )
        return source
