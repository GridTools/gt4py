# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import abc
import numbers
import os
import textwrap
from dataclasses import dataclass, field
from inspect import getdoc
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jinja2
import numpy

from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.definitions import AccessKind, DomainInfo, FieldInfo, ParameterInfo
from gtc import gtir
from gtc.passes.gtir_legacy_extents import compute_legacy_extents
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.utils import dimension_flags_to_names


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


@dataclass
class ModuleData:
    field_info: Dict[str, Optional[FieldInfo]] = field(default_factory=dict)
    parameter_info: Dict[str, Optional[ParameterInfo]] = field(default_factory=dict)
    unreferenced: List[str] = field(default_factory=list)

    @property
    def field_names(self):
        return self.field_info.keys()

    @property
    def parameter_names(self):
        return self.parameter_info.keys()


def make_args_data_from_iir(implementation_ir: gt_ir.StencilImplementation) -> ModuleData:
    data = ModuleData()

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
            access = AccessKind.READ_WRITE if arg.name in out_fields else AccessKind.READ_ONLY
            if arg.name not in implementation_ir.unreferenced:
                field_decl = implementation_ir.fields[arg.name]
                data.field_info[arg.name] = FieldInfo(
                    access=access,
                    boundary=implementation_ir.fields_extents[arg.name].to_boundary(),
                    axes=tuple(field_decl.axes),
                    data_dims=tuple(field_decl.data_dims),
                    dtype=field_decl.data_type.dtype,
                )
            else:
                data.field_info[arg.name] = None
        else:
            if arg.name not in implementation_ir.unreferenced:
                data.parameter_info[arg.name] = ParameterInfo(
                    dtype=implementation_ir.parameters[arg.name].data_type.dtype
                )
            else:
                data.parameter_info[arg.name] = None

    data.unreferenced = implementation_ir.unreferenced

    return data


def get_unused_params_from_gtir(
    pipeline: GtirPipeline,
) -> List[Union[gtir.FieldDecl, gtir.ScalarDecl]]:
    node = pipeline.gtir
    field_names = {
        param.name for param in node.params if isinstance(param, (gtir.FieldDecl, gtir.ScalarDecl))
    }
    used_field_names = (
        node.iter_tree().if_isinstance(gtir.FieldAccess, gtir.ScalarAccess).getattr("name").to_set()
    )
    return [
        param for param in node.params if param.name in field_names.difference(used_field_names)
    ]


def make_args_data_from_gtir(pipeline: GtirPipeline) -> ModuleData:
    data = ModuleData()
    node = pipeline.full()
    field_extents = compute_legacy_extents(node)

    write_fields = (
        node.iter_tree()
        .if_isinstance(gtir.ParAssignStmt)
        .getattr("left")
        .if_isinstance(gtir.FieldAccess)
        .getattr("name")
        .to_list()
    )

    referenced_field_params = [
        param.name for param in node.params if isinstance(param, gtir.FieldDecl)
    ]
    for name in sorted(referenced_field_params):
        data.field_info[name] = FieldInfo(
            access=AccessKind.READ_WRITE if name in write_fields else AccessKind.READ_ONLY,
            boundary=field_extents[name].to_boundary(),
            axes=tuple(dimension_flags_to_names(node.symtable_[name].dimensions).upper()),
            data_dims=tuple(node.symtable_[name].data_dims),
            dtype=numpy.dtype(node.symtable_[name].dtype.name.lower()),
        )

    referenced_scalar_params = [
        param.name for param in node.params if param.name not in referenced_field_params
    ]
    for name in sorted(referenced_scalar_params):
        data.parameter_info[name] = ParameterInfo(
            dtype=numpy.dtype(node.symtable_[name].dtype.name.lower())
        )

    unref_params = get_unused_params_from_gtir(pipeline)
    for param in sorted(unref_params, key=lambda decl: decl.name):
        if isinstance(param, gtir.FieldDecl):
            data.field_info[param.name] = None
        elif isinstance(param, gtir.ScalarDecl):
            data.parameter_info[param.name] = None

    data.unreferenced = [*sorted(param.name for param in unref_params)]
    return data


class BaseModuleGenerator(abc.ABC):

    SOURCE_LINE_LENGTH = 120
    TEMPLATE_INDENT_SIZE = 4
    DOMAIN_ARG_NAME = "_domain_"
    ORIGIN_ARG_NAME = "_origin_"
    SPLITTERS_NAME = "_splitters_"

    TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "stencil_module.py.in")

    _builder: Optional["StencilBuilder"]
    args_data: ModuleData
    template: jinja2.Template

    def __init__(self, builder: Optional["StencilBuilder"] = None):
        self._builder = builder
        self.args_data = ModuleData()
        with open(self.TEMPLATE_PATH, "r") as f:
            self.template = jinja2.Template(f.read())

    def __call__(
        self,
        args_data: ModuleData,
        builder: Optional["StencilBuilder"] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate source code for a Python module containing a StencilObject.

        A possible reaosn for extending is processing additional kwargs,
        using a different template might require completely overriding.
        """
        if builder:
            self._builder = builder
        self.args_data = args_data

        module_source = self.template.render(
            imports=self.generate_imports(),
            module_members=self.generate_module_members(),
            class_name=self.generate_class_name(),
            class_members=self.generate_class_members(),
            docstring=self.generate_docstring(),
            gt_backend=self.generate_backend_name(),
            gt_source=self.generate_sources(),
            gt_domain_info=self.generate_domain_info(),
            gt_field_info=repr(self.args_data.field_info),
            gt_parameter_info=repr(self.args_data.parameter_info),
            gt_constants=self.generate_constants(),
            gt_options=self.generate_options(),
            stencil_signature=self.generate_signature(),
            field_names=self.args_data.field_names,
            param_names=self.args_data.parameter_names,
            pre_run=self.generate_pre_run(),
            post_run=self.generate_post_run(),
            implementation=self.generate_implementation(),
        )
        if self.builder.options.as_dict()["format_source"]:
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
        """Generate the work code inside the stencil object's run function."""
        pass

    def generate_imports(self) -> str:
        """Generate import statements and related code for the stencil class module."""
        return ""

    def generate_class_name(self) -> str:
        """
        Generate the name of the stencil class.

        This should ususally be deferred to the chosen caching strategy via
        the builder object (see default implementation).
        """
        return self.builder.class_name

    def generate_docstring(self) -> str:
        """
        Generate the docstring of the stencil object.

        The default is to return the stencil definition's docstring or an
        empty string.
        The output should be least based on the stencil definition's docstring,
        if one exists.
        """
        return getdoc(self.builder.definition) or ""

    def generate_backend_name(self) -> str:
        """
        Return the name of the backend.

        There should never be a need to override this.
        """
        return self.backend_name

    def generate_sources(self) -> Dict[str, str]:
        """
        Return the source code of the stencil definition in string format.

        This is unlikely to require overriding.
        """
        if self.builder.definition_ir.sources is not None:
            return {
                key: gt_utils.text.format_source(value, line_length=self.SOURCE_LINE_LENGTH)
                for key, value in self.builder.definition_ir.sources
            }
        return {}

    def generate_constants(self) -> Dict[str, str]:
        """
        Return a mapping of named numeric constants passed as externals.

        This is unlikely to require overriding.
        """
        if self.builder.definition_ir.externals:
            return {
                name: repr(value)
                for name, value in self.builder.definition_ir.externals.items()
                if isinstance(value, numbers.Number)
            }
        return {}

    def generate_options(self) -> Dict[str, Any]:
        """
        Return dictionary of build options.

        Must exclude options that should never be cached.
        """
        return {
            key: value
            for key, value in self.builder.options.as_dict().items()
            if key not in ["build_info"]
        }

    def generate_domain_info(self) -> str:
        """
        Generate a ``DomainInfo`` constructor call with the correct arguments.

        Might require overriding for module generators of non-cartesian backends.
        """
        parallel_axes = self.builder.definition_ir.domain.parallel_axes or []
        sequential_axis = self.builder.definition_ir.domain.sequential_axis.name
        domain_info = repr(
            DomainInfo(
                parallel_axes=tuple(ax.name for ax in parallel_axes),
                sequential_axis=sequential_axis,
                ndim=len(parallel_axes) + (1 if sequential_axis else 0),
            )
        )
        return domain_info

    def generate_module_members(self) -> str:
        """
        Generate additional module level code after all imports.

        May contain any executable module level code including function and class defs.
        """
        return ""

    def generate_class_members(self) -> str:
        """
        Generate additional stencil class members.

        May contain any class level code including methods.
        """
        return ""

    def generate_signature(self) -> str:
        """
        Generate the stencil definition specific part of the stencil object's ``__call__`` signature.

        Unlikely to require overriding.
        """
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
        """Additional code to be run just before the run method (implementation) is called."""
        return ""

    def generate_post_run(self) -> str:
        """Additional code to be run just after the run method (implementation) is called."""
        return ""


def iir_is_not_emtpy(implementation_ir: gt_ir.StencilImplementation) -> bool:
    return bool(implementation_ir.multi_stages)


def gtir_is_not_emtpy(pipeline: GtirPipeline) -> bool:
    node = pipeline.full()
    return bool(node.iter_tree().if_isinstance(gtir.ParAssignStmt).to_list())


def iir_has_effect(implementation_ir: gt_ir.StencilImplementation) -> bool:
    return bool(implementation_ir.has_effect)


def gtir_has_effect(pipeline: GtirPipeline) -> bool:
    return True


class PyExtModuleGenerator(BaseModuleGenerator):
    """
    Module Generator for use with backends that generate c++ python extensions.

    Will either use ImplementationIR or GTIR depending on the backend's USE_LEGACY_TOOLCHAIN
    class attribute. Using with other IRs requires subclassing and overriding ``_is_not_empty()``
    and ``_has_effect()`` methods.
    """

    pyext_module_name: str
    pyext_file_path: str

    def __init__(self):
        super().__init__()
        self.pyext_module_name = ""
        self.pyext_file_path = ""

    def __call__(
        self,
        args_data: ModuleData,
        builder: Optional["StencilBuilder"] = None,
        **kwargs: Any,
    ) -> str:
        self.pyext_module_name = kwargs["pyext_module_name"]
        self.pyext_file_path = kwargs["pyext_file_path"]
        return super().__call__(args_data, builder, **kwargs)

    def _is_not_empty(self) -> bool:
        if self.builder.backend.USE_LEGACY_TOOLCHAIN:
            return iir_is_not_emtpy(self.builder.implementation_ir)
        return gtir_is_not_emtpy(self.builder.gtir_pipeline)

    def generate_imports(self) -> str:
        source = ["from gt4py import utils as gt_utils"]
        if self._is_not_empty:
            source.append(
                textwrap.dedent(
                    f"""
                pyext_module = gt_utils.make_module_from_file(
                    "{self.pyext_module_name}", "{self.pyext_file_path}", public_import=True
                )
                """
                )
            )
        return "\n".join(source)

    def _has_effect(self) -> bool:
        if self.builder.backend.USE_LEGACY_TOOLCHAIN:
            return iir_has_effect(self.builder.implementation_ir)
        return gtir_has_effect(self.builder.gtir_pipeline)

    def generate_implementation(self) -> str:
        definition_ir = self.builder.definition_ir
        sources = gt_utils.text.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        api_fields = set(field.name for field in definition_ir.api_fields)
        for arg in definition_ir.api_signature:
            if arg.name not in self.args_data.unreferenced:
                args.append(arg.name)
                if arg.name in api_fields:
                    args.append("list(_origin_['{}'])".format(arg.name))

        # only generate implementation if any multi_stages are present. e.g. if no statement in the
        # stencil has any effect on the API fields, this may not be the case since they could be
        # pruned.
        if self._has_effect():
            source = textwrap.dedent(
                f"""
                # Load or generate a GTComputation object for the current domain size
                pyext_module.run_computation({",".join(["list(_domain_)", *args, "exec_info"])})
                """
            )
            sources.extend(source.splitlines())
        else:
            sources.extend("\n")

        return sources.text


class CUDAPyExtModuleGenerator(PyExtModuleGenerator):
    def generate_implementation(self) -> str:
        source = super().generate_implementation()
        if self.builder.options.backend_opts.get("device_sync", True):
            source += textwrap.dedent(
                """
                    cupy.cuda.Device(0).synchronize()
                """
            )
        return source

    def generate_imports(self) -> str:
        source = (
            textwrap.dedent(
                """
                import cupy
                """
            )
            + super().generate_imports()
        )
        return source
