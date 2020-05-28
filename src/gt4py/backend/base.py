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
import pickle
import sys
import types
from typing import Any, Dict, List, Optional, Type

import jinja2

from gt4py import analysis as gt_analysis
from gt4py import config as gt_config
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from . import pyext_builder


REGISTRY = gt_utils.Registry()


def from_name(name: str):
    return REGISTRY.get(name, None)


def register(backend_cls):
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
    #: Backend name: str
    name = None

    #: Backend-specific options:
    #:  Dict[name: str, info: Dict[str, Any]]]
    #:      + info:
    #:          - versioning: bool
    #:          - description [optional]: str
    options = None

    #: Backend-specific storage parametrization: Dict[str, Any]
    #:  - "alignment": int (in bytes)
    #:  - "device": str ("cpu" | "gpu")
    #:  - "layout_map": Tuple[bool] -> Tuple[Union[int, None]]
    #:  - "is_compatible_layout": StorageLikeInstance -> bool
    #:  - "is_compatible_type": StorageLikeInstance -> bool
    storage_info = None

    @classmethod
    def get_options_id(cls, options: gt_definitions.BuildOptions):
        filtered_options = copy.deepcopy(options)
        id_names = set(name for name, info in cls.options.items() if info["versioning"])
        non_id_names = set(filtered_options.backend_opts.keys()) - id_names
        for name in non_id_names:
            del filtered_options.backend_opts[name]

        return filtered_options.shashed_id

    @classmethod
    @abc.abstractmethod
    def load(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
    ):
        pass

    @classmethod
    @abc.abstractmethod
    def generate(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
    ):
        pass


class BaseBackend(Backend):

    MODULE_GENERATOR_CLASS = None

    @classmethod
    def get_stencil_package_name(cls, stencil_id: gt_definitions.StencilID):
        components = stencil_id.qualified_name.split(".")
        package_name = ".".join([gt_config.code_settings["root_package_name"]] + components[:-1])
        return package_name

    @classmethod
    def get_stencil_module_name(
        cls, stencil_id: gt_definitions.StencilID, *, qualified: bool = False
    ):
        module_name = "m_{}".format(cls.get_stencil_class_name(stencil_id))
        if qualified:
            module_name = "{}.{}".format(cls.get_stencil_package_name(stencil_id), module_name)
        return module_name

    @classmethod
    def get_stencil_class_name(cls, stencil_id: gt_definitions.StencilID):
        components = stencil_id.qualified_name.split(".")
        class_name = "{name}__{backend}_{id}".format(
            name=components[-1], backend=gt_utils.slugify(cls.name), id=stencil_id.version
        )
        return class_name

    @classmethod
    def get_base_path(cls):
        # Initialize cache folder
        cache_root = os.path.join(
            gt_config.cache_settings["root_path"], gt_config.cache_settings["dir_name"]
        )
        if not os.path.exists(cache_root):
            gt_utils.make_dir(cache_root, is_cache=True)

        cpython_id = "py{major}{minor}_{api}".format(
            major=sys.version_info.major, minor=sys.version_info.minor, api=sys.api_version
        )
        base_path = os.path.join(cache_root, cpython_id, gt_utils.slugify(cls.name))
        return base_path

    @classmethod
    def get_stencil_package_path(cls, stencil_id: gt_definitions.StencilID):
        components = stencil_id.qualified_name.split(".")
        path = os.path.join(cls.get_base_path(), *components[:-1])
        return path

    @classmethod
    def get_stencil_module_path(cls, stencil_id: gt_definitions.StencilID):
        components = stencil_id.qualified_name.split(".")
        path = os.path.join(
            cls.get_base_path(), *components[:-1], cls.get_stencil_module_name(stencil_id) + ".py"
        )
        return path

    @classmethod
    def get_cache_info_path(cls, stencil_id: gt_definitions.StencilID):
        path = str(cls.get_stencil_module_path(stencil_id))[:-3] + ".cacheinfo"
        return path

    @classmethod
    def generate_cache_info(
        cls, stencil_id: gt_definitions.StencilID, extra_cache_info: Dict[str, Any]
    ):
        module_file_name = cls.get_stencil_module_path(stencil_id)
        with open(module_file_name, "r") as f:
            source = f.read()
        cache_info = {
            # "gt4py_version": 0.5,
            "backend": cls.name,
            "stencil_name": stencil_id.qualified_name,
            "stencil_version": stencil_id.version,
            "module_shash": gt_utils.shash(source),
            **extra_cache_info,
        }

        return cache_info

    @classmethod
    def update_cache(cls, stencil_id: gt_definitions.StencilID, extra_cache_info: Dict[str, Any]):
        cache_info = cls.generate_cache_info(stencil_id, extra_cache_info)
        cache_file_name = cls.get_cache_info_path(stencil_id)
        os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
        with open(cache_file_name, "wb") as f:
            pickle.dump(cache_info, f)

    @classmethod
    def validate_cache_info(
        cls,
        stencil_id: gt_definitions.StencilID,
        cache_info: Dict[str, Any],
        *,
        validate_hash: bool = True,
    ):
        result = True
        try:
            cache_info = types.SimpleNamespace(**cache_info)

            module_file_name = cls.get_stencil_module_path(stencil_id)
            with open(module_file_name, "r") as f:
                source = f.read()
            module_shash = gt_utils.shash(source)

            if validate_hash:
                result = (
                    cache_info.backend == cls.name
                    and cache_info.stencil_name == stencil_id.qualified_name
                    and cache_info.stencil_version == stencil_id.version
                    and cache_info.module_shash == module_shash
                )

        except Exception:
            result = False

        return result

    @classmethod
    def check_cache(cls, stencil_id: gt_definitions.StencilID, *, validate_hash: bool = True):
        try:
            cache_file_name = cls.get_cache_info_path(stencil_id)
            with open(cache_file_name, "rb") as f:
                cache_info = pickle.load(f)
            assert isinstance(cache_info, dict)
            result = cls.validate_cache_info(stencil_id, cache_info, validate_hash=validate_hash)

        except Exception:
            result = False

        return result

    @classmethod
    def _check_options(cls, options: gt_definitions.BuildOptions):
        assert cls.options is not None
        unknown_options = set(options.backend_opts.keys()) - set(cls.options.keys())
        if unknown_options:
            raise ValueError("Unknown backend options: '{}'".format(unknown_options))

    @classmethod
    def _load(cls, stencil_id: gt_definitions.StencilID, definition_func: types.FunctionType):
        stencil_class_name = cls.get_stencil_class_name(stencil_id)
        file_name = cls.get_stencil_module_path(stencil_id)
        stencil_module = gt_utils.make_module_from_file(stencil_class_name, file_name)
        stencil_class = getattr(stencil_module, stencil_class_name)
        stencil_class.__module__ = cls.get_stencil_module_name(stencil_id, qualified=True)
        stencil_class._gt_id_ = stencil_id.version
        stencil_class.definition_func = staticmethod(definition_func)

        return stencil_class

    @classmethod
    def load(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
    ):
        stencil_class = None
        if stencil_id is not None:
            cls._check_options(options)
            validate_hash = options.dev_opts.get("cache-validation", True)
            if cls.check_cache(stencil_id, validate_hash=validate_hash):
                stencil_class = cls._load(stencil_id, definition_func)

        return stencil_class

    @classmethod
    def _generate_module(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
        *,
        extra_cache_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        file_name = cls.get_stencil_module_path(stencil_id)
        generator = cls.MODULE_GENERATOR_CLASS(cls)
        module_source = generator(stencil_id, definition_ir, options, **kwargs)

        if options.dev_opts.get("code-generation", True):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "w") as f:
                f.write(module_source)
            extra_cache_info = extra_cache_info or {}
            cls.update_cache(stencil_id, extra_cache_info)

        return cls._load(stencil_id, definition_func)

    @classmethod
    def generate(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
    ):
        cls._check_options(options)
        return cls._generate_module(stencil_id, definition_ir, definition_func, options)


class BasePyExtBackend(BaseBackend):
    @classmethod
    def get_pyext_module_name(cls, stencil_id: gt_definitions.StencilID, *, qualified=False):
        module_name = cls.get_stencil_module_name(stencil_id, qualified=qualified) + "_pyext"
        return module_name

    @classmethod
    def get_pyext_class_name(cls, stencil_id: gt_definitions.StencilID):
        module_name = cls.get_stencil_class_name(stencil_id) + "_pyext"
        return module_name

    @classmethod
    def get_pyext_build_path(cls, stencil_id: gt_definitions.StencilID):
        path = os.path.join(
            cls.get_stencil_package_path(stencil_id),
            cls.get_pyext_module_name(stencil_id) + "_BUILD",
        )

        return path

    @classmethod
    def generate_cache_info(
        cls, stencil_id: gt_definitions.StencilID, extra_cache_info: Dict[str, Any]
    ):
        cache_info = super(BasePyExtBackend, cls).generate_cache_info(stencil_id, {})

        cache_info["pyext_file_path"] = extra_cache_info["pyext_file_path"]
        cache_info["pyext_md5"] = hashlib.md5(
            open(cache_info["pyext_file_path"], "rb").read()
        ).hexdigest()

        return cache_info

    @classmethod
    def validate_cache_info(
        cls,
        stencil_id: gt_definitions.StencilID,
        cache_info: Dict[str, Any],
        *,
        validate_hash: bool = True,
    ):
        result = True
        try:
            assert super(BasePyExtBackend, cls).validate_cache_info(
                stencil_id, cache_info, validate_hash=validate_hash
            )
            pyext_md5 = hashlib.md5(open(cache_info["pyext_file_path"], "rb").read()).hexdigest()
            if validate_hash:
                result = pyext_md5 == cache_info["pyext_md5"]

        except Exception:
            result = False

        return result

    @classmethod
    def build_extension_module(
        cls,
        stencil_id: gt_definitions.StencilID,
        pyext_sources: Dict[str, str],
        pyext_build_opts: Dict[str, str],
        *,
        pyext_extra_include_dirs: List[str] = None,
        uses_cuda: bool = False,
    ):

        # Build extension module
        pyext_build_path = os.path.relpath(cls.get_pyext_build_path(stencil_id))
        os.makedirs(pyext_build_path, exist_ok=True)
        sources = []
        for key, source in pyext_sources.items():
            src_file_name = os.path.join(pyext_build_path, key)
            src_ext = src_file_name.split(".")[-1]
            if src_ext not in ["h", "hpp"]:
                sources.append(src_file_name)

            if source is not gt_utils.NOTHING:
                with open(src_file_name, "w") as f:
                    f.write(source)

        pyext_target_path = cls.get_stencil_package_path(stencil_id)
        qualified_pyext_name = cls.get_pyext_module_name(stencil_id, qualified=True)

        if uses_cuda:
            module_name, file_path = pyext_builder.build_gtcuda_ext(
                qualified_pyext_name,
                sources=sources,
                build_path=pyext_build_path,
                target_path=pyext_target_path,
                extra_include_dirs=pyext_extra_include_dirs,
                **pyext_build_opts,
            )
        else:
            module_name, file_path = pyext_builder.build_gtcpu_ext(
                qualified_pyext_name,
                sources=sources,
                build_path=pyext_build_path,
                target_path=pyext_target_path,
                extra_include_dirs=pyext_extra_include_dirs,
                **pyext_build_opts,
            )
        assert module_name == qualified_pyext_name

        return module_name, file_path

    @classmethod
    @abc.abstractmethod
    def generate(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
    ):
        pass


class BaseModuleGenerator(abc.ABC):

    SOURCE_LINE_LENGTH = 120
    TEMPLATE_INDENT_SIZE = 4
    DOMAIN_ARG_NAME = "_domain_"
    ORIGIN_ARG_NAME = "_origin_"
    SPLITTERS_NAME = "_splitters_"

    TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "stencil_module.py.in")

    def __init__(self, backend_class: Type[Backend]):
        assert issubclass(backend_class, BaseBackend)
        self.backend_class = backend_class
        self.options = None
        self.stencil_id = None
        self.definition_ir = None
        with open(self.TEMPLATE_PATH, "r") as f:
            self.template = jinja2.Template(f.read())

        self.implementation_ir = None
        self.wrapper_info = None

    def __call__(
        self,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        options: gt_definitions.BuildOptions,
        **kwargs,
    ):
        self.options = options
        self.stencil_id = stencil_id
        self.definition_ir = definition_ir

        # IIR is generated by default using the internal analysis pipeline,
        # but if a backend uses its own pipeline it can pass it explicitly to
        # the module generator using kwargs
        self.implementation_ir = (
            kwargs["implementation_ir"]
            if "implementation_ir" in kwargs
            else self._generate_implementation_ir()
        )
        # By default the wrapper_info is generated from the IIR, but
        # if a backend generates it using other mechanisms it can pass it
        # explicitly to the module generator using kwargs
        self.wrapper_info = (
            kwargs["wrapper_info"] if "wrapper_info" in kwargs else self._generate_wrapper_info()
        )

        wrapper_info = self.wrapper_info
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
            class_name=self.stencil_class_name,
            class_members=class_members,
            docstring=wrapper_info["docstring"],
            gt_backend=self.backend_name,
            gt_source=wrapper_info["sources"],
            gt_domain_info=wrapper_info["domain_info"],
            gt_field_info=repr(wrapper_info["field_info"]),
            gt_parameter_info=repr(wrapper_info["parameter_info"]),
            gt_constants=wrapper_info["gt_constants"],
            gt_options=wrapper_info["gt_options"],
            stencil_signature=stencil_signature,
            field_names=wrapper_info["field_info"].keys(),
            param_names=wrapper_info["parameter_info"].keys(),
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

    @property
    def stencil_class_name(self) -> str:
        return self.backend_class.get_stencil_class_name(self.stencil_id)

    def _generate_implementation_ir(self) -> gt_ir.StencilImplementation:
        implementation_ir = gt_analysis.transform(self.definition_ir, self.options)
        return implementation_ir

    def _generate_wrapper_info(self) -> Dict[str, Any]:
        info = {}
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

        info["field_info"] = field_info = {}
        info["parameter_info"] = parameter_info = {}

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
    def __init__(self, backend_class: Type[Backend]):
        super().__init__(backend_class)
        self.pyext_module_name = None
        self.pyext_file_path = None

    def __call__(
        self,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        options: gt_definitions.BuildOptions,
        **kwargs,
    ):
        self.pyext_module_name = kwargs["pyext_module_name"]
        self.pyext_file_path = kwargs["pyext_file_path"]
        return super().__call__(stencil_id, definition_ir, options, **kwargs)

    def generate_imports(self) -> str:
        source = """
from gt4py import utils as gt_utils

pyext_module = gt_utils.make_module_from_file("{pyext_module_name}", "{pyext_file_path}", public_import=True)
        """.format(
            pyext_module_name=self.pyext_module_name, pyext_file_path=self.pyext_file_path
        )

        return source

    def generate_implementation(self) -> str:
        sources = gt_utils.text.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        api_fields = set(field.name for field in self.definition_ir.api_fields)
        for arg in self.definition_ir.api_signature:
            if arg.name not in self.wrapper_info["unreferenced"]:
                args.append(arg.name)
                if arg.name in api_fields:
                    args.append("list(_origin_['{}'])".format(arg.name))

        source = """
# Load or generate a GTComputation object for the current domain size
pyext_module.run_computation(list(_domain_), {run_args}, exec_info)
""".format(
            run_args=", ".join(args)
        )
        sources.extend(source.splitlines())

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
