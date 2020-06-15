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
import numbers
import os
import pickle
import sys
import types

import jinja2

from gt4py import analysis as gt_analysis
from gt4py import config as gt_config
from gt4py import definitions as gt_definitions
from gt4py import utils as gt_utils
from gt4py import ir as gt_ir


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
    name = None

    options = None
    # Dict[str, info: Dict[str, Any]]]
    #   + info:
    #       - versioning: bool
    #       - description [optional]: str

    @classmethod
    def get_options_id(cls, options):
        filtered_options = copy.deepcopy(options)
        id_names = set(name for name, info in cls.options.items() if info["versioning"])
        non_id_names = set(filtered_options.backend_opts.keys()) - id_names
        for name in non_id_names:
            del filtered_options.backend_opts[name]

        return filtered_options.shashed_id

    @classmethod
    @abc.abstractmethod
    def load(cls, stencil_id, definition_func, options):
        pass

    @classmethod
    @abc.abstractmethod
    def generate(cls, stencil_id, definition_ir, definition_func, options):
        pass


class BaseBackend(Backend):

    GENERATOR_CLASS = None

    @classmethod
    def get_stencil_package_name(cls, stencil_id):
        components = stencil_id.qualified_name.split(".")
        package_name = ".".join([gt_config.code_settings["root_package_name"]] + components[:-1])
        return package_name

    @classmethod
    def get_stencil_module_name(cls, stencil_id, *, qualified=False):
        module_name = "m_{}".format(cls.get_stencil_class_name(stencil_id))
        if qualified:
            module_name = "{}.{}".format(cls.get_stencil_package_name(stencil_id), module_name)
        return module_name

    @classmethod
    def get_stencil_class_name(cls, stencil_id):
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
    def get_stencil_package_path(cls, stencil_id):
        components = stencil_id.qualified_name.split(".")
        path = os.path.join(cls.get_base_path(), *components[:-1])
        return path

    @classmethod
    def get_stencil_module_path(cls, stencil_id):
        components = stencil_id.qualified_name.split(".")
        path = os.path.join(
            cls.get_base_path(), *components[:-1], cls.get_stencil_module_name(stencil_id) + ".py"
        )
        return path

    @classmethod
    def get_cache_info_path(cls, stencil_id):
        path = str(cls.get_stencil_module_path(stencil_id))[:-3] + ".cacheinfo"
        return path

    @classmethod
    def generate_cache_info(cls, stencil_id, extra_cache_info):
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
    def update_cache(cls, stencil_id, extra_cache_info):
        cache_info = cls.generate_cache_info(stencil_id, extra_cache_info)
        cache_file_name = cls.get_cache_info_path(stencil_id)
        os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
        with open(cache_file_name, "wb") as f:
            pickle.dump(cache_info, f)

    @classmethod
    def validate_cache_info(cls, stencil_id, cache_info):
        try:
            cache_info = types.SimpleNamespace(**cache_info)

            module_file_name = cls.get_stencil_module_path(stencil_id)
            with open(module_file_name, "r") as f:
                source = f.read()
            module_shash = gt_utils.shash(source)

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
    def check_cache(cls, stencil_id):
        try:
            cache_file_name = cls.get_cache_info_path(stencil_id)
            with open(cache_file_name, "rb") as f:
                cache_info = pickle.load(f)
            assert isinstance(cache_info, dict)
            result = cls.validate_cache_info(stencil_id, cache_info)

        except Exception:
            result = False

        return result

    @classmethod
    def _check_options(cls, options):
        assert cls.options is not None
        unknown_options = set(options.backend_opts.keys()) - set(cls.options.keys())
        if unknown_options:
            raise ValueError("Unknown backend options: '{}'".format(unknown_options))

    @classmethod
    def _load(cls, stencil_id, definition_func):
        stencil_class_name = cls.get_stencil_class_name(stencil_id)
        file_name = cls.get_stencil_module_path(stencil_id)
        stencil_module = gt_utils.make_module_from_file(stencil_class_name, file_name)
        stencil_class = getattr(stencil_module, stencil_class_name)
        stencil_class.__module__ = cls.get_stencil_module_name(stencil_id, qualified=True)
        stencil_class._gt_id_ = stencil_id.version
        stencil_class.definition_func = staticmethod(definition_func)

        return stencil_class

    @classmethod
    def load(cls, stencil_id, definition_func, options):
        stencil_class = None
        if stencil_id is not None:
            cls._check_options(options)
            if cls.check_cache(stencil_id):
                stencil_class = cls._load(stencil_id, definition_func)

        return stencil_class

    @classmethod
    def _generate_module(
        cls, stencil_id, implementation_ir, definition_func, generator_options, extra_cache_info
    ):
        generator = cls.GENERATOR_CLASS(cls, options=generator_options)
        module_source = generator(stencil_id, implementation_ir)

        file_name = cls.get_stencil_module_path(stencil_id)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(module_source)
        cls.update_cache(stencil_id, extra_cache_info)

        return cls._load(stencil_id, definition_func)

    @classmethod
    def generate(cls, stencil_id, definition_ir, definition_func, options):
        cls._check_options(options)
        implementation_ir = gt_analysis.transform(definition_ir, options)
        return cls._generate_module(
            stencil_id, implementation_ir, definition_func, copy.deepcopy(options.as_dict()), {}
        )


class BaseModuleGenerator(abc.ABC):

    SOURCE_LINE_LENGTH = 120
    TEMPLATE_INDENT_SIZE = 4
    DOMAIN_ARG_NAME = "_domain_"
    ORIGIN_ARG_NAME = "_origin_"
    SPLITTERS_NAME = "_splitters_"

    TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "stencil_module.py.in")

    def __init__(self, backend_class, options):
        assert issubclass(backend_class, BaseBackend)
        self.backend_class = backend_class
        self.options = types.SimpleNamespace(**options)
        self.stencil_id = None
        self.implementation_ir = None
        with open(self.TEMPLATE_PATH, "r") as f:
            self.template = jinja2.Template(f.read())

    def __call__(self, stencil_id, implementation_ir):
        self.stencil_id = stencil_id
        self.implementation_ir = implementation_ir

        stencil_signature = self.generate_signature()

        sources = {}
        if implementation_ir.sources is not None:
            sources = {
                key: gt_utils.text.format_source(value, line_length=self.SOURCE_LINE_LENGTH)
                for key, value in implementation_ir.sources
            }

        parallel_axes = implementation_ir.domain.parallel_axes or []
        sequential_axis = implementation_ir.domain.sequential_axis.name
        domain_info = repr(
            gt_definitions.DomainInfo(
                parallel_axes=tuple(ax.name for ax in parallel_axes),
                sequential_axis=sequential_axis,
                ndims=len(parallel_axes) + (1 if sequential_axis else 0),
            )
        )

        field_info = {}
        field_names = []
        parameter_info = {}
        param_names = []

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
                field_names.append(arg.name)
            else:
                if arg.name not in implementation_ir.unreferenced:
                    parameter_info[arg.name] = gt_definitions.ParameterInfo(
                        dtype=implementation_ir.parameters[arg.name].data_type.dtype
                    )
                else:
                    parameter_info[arg.name] = None
                param_names.append(arg.name)

        field_info = repr(field_info)
        parameter_info = repr(parameter_info)

        if implementation_ir.externals:
            gt_constants = {
                name: repr(value)
                for name, value in implementation_ir.externals.items()
                if isinstance(value, numbers.Number)
            }
        else:
            gt_constants = {}

        gt_options = dict(self.options.__dict__)
        if "build_info" in gt_options:
            del gt_options["build_info"]

        # Concrete implementation in the subclasses
        imports = self.generate_imports()
        module_members = self.generate_module_members()
        class_members = self.generate_class_members()
        implementation = self.generate_implementation()

        module_source = self.template.render(
            imports=imports,
            module_members=module_members,
            class_name=self.stencil_class_name,
            class_members=class_members,
            docstring=implementation_ir.docstring,
            gt_backend=self.backend_name,
            gt_source=sources,
            gt_domain_info=domain_info,
            gt_field_info=field_info,
            gt_parameter_info=parameter_info,
            gt_constants=gt_constants,
            gt_options=gt_options,
            stencil_signature=stencil_signature,
            field_names=field_names,
            param_names=param_names,
            synchronization=self.generate_synchronization(
                [
                    k
                    for k in implementation_ir.fields.keys()
                    if k not in implementation_ir.temporary_fields
                    and k not in implementation_ir.unreferenced
                ]
            ),
            mark_modified=self.generate_mark_modified(
                [
                    k
                    for k in out_fields
                    if k not in implementation_ir.temporary_fields
                    and k not in implementation_ir.unreferenced
                ]
            ),
            implementation=implementation,
        )
        module_source = gt_utils.text.format_source(
            module_source, line_length=self.SOURCE_LINE_LENGTH
        )

        return module_source

    @property
    def backend_name(self):
        return self.backend_class.name

    @property
    def stencil_class_name(self):
        return self.backend_class.get_stencil_class_name(self.stencil_id)

    def generate_synchronization(self, field_names):
        return ""

    def generate_mark_modified(self, output_field_names):
        return ""

    def generate_signature(self):
        args = []
        keyword_args = ["*"]
        for arg in self.implementation_ir.api_signature:
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

    def generate_imports(self):
        source = ""
        return source

    def generate_module_members(self):
        source = ""
        return source

    def generate_class_members(self):
        source = ""
        return source

    @abc.abstractmethod
    def generate_implementation(self):
        pass
