# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, cast

import jinja2
import numpy

from gt4py import utils as gt_utils
from gt4py.definitions import AccessKind, DomainInfo, FieldInfo, ParameterInfo, StencilID
from gtc import gtir, gtir_to_oir
from gtc.definitions import Boundary
from gtc.passes.gtir_k_boundary import compute_k_boundary, compute_min_k_size
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_access_kinds import compute_access_kinds
from gtc.passes.oir_optimizations.utils import compute_fields_extents
from gtc.utils import dimension_flags_to_names


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


@dataclass
class ModuleData:
    domain_info: Optional[DomainInfo] = None
    field_info: Dict[str, FieldInfo] = field(default_factory=dict)
    parameter_info: Dict[str, ParameterInfo] = field(default_factory=dict)
    unreferenced: List[str] = field(default_factory=list)

    @property
    def field_names(self) -> Set[str]:
        """Set of all field names."""
        return set(self.field_info.keys())

    @property
    def parameter_names(self) -> Set[str]:
        """Set of all parameter names."""
        return set(self.parameter_info.keys())


_args_data_cache: Dict[StencilID, ModuleData] = {}


def make_args_data_from_gtir(pipeline: GtirPipeline) -> ModuleData:
    """
    Compute module data containing information about stencil arguments from gtir.

    This is no longer compatible with the legacy backends.
    """
    if pipeline.stencil_id in _args_data_cache:
        return _args_data_cache[pipeline.stencil_id]
    data = ModuleData()

    # NOTE: pipeline.gtir has not had prune_unused_parameters applied.
    all_params = pipeline.gtir.params

    node = pipeline.full()
    oir = gtir_to_oir.GTIRToOIR().visit(node)
    field_extents = compute_fields_extents(oir)
    accesses = compute_access_kinds(oir)

    min_sequential_axis_size = compute_min_k_size(node)
    data.domain_info = DomainInfo(
        parallel_axes=("I", "J"),
        sequential_axis="K",
        min_sequential_axis_size=min_sequential_axis_size,
        ndim=3,
    )

    for field_decl in (param for param in all_params if isinstance(param, gtir.FieldDecl)):
        access = accesses[field_decl.name]
        dtype = numpy.dtype(field_decl.dtype.name.lower())

        if access != AccessKind.NONE:
            k_boundary = compute_k_boundary(node)[field_decl.name]
            boundary = Boundary(*field_extents[field_decl.name].to_boundary()[0:2], k_boundary)
        else:
            boundary = Boundary.zeros(ndims=3)

        data.field_info[str(field_decl.name)] = FieldInfo(
            access=access,
            boundary=boundary,
            axes=tuple(dimension_flags_to_names(field_decl.dimensions).upper()),
            data_dims=tuple(field_decl.data_dims),
            dtype=dtype,
        )

    for scalar_decl in (param for param in all_params if isinstance(param, gtir.ScalarDecl)):
        access = cast(Literal[AccessKind.NONE, AccessKind.READ], accesses[scalar_decl.name])
        assert access in {AccessKind.NONE, AccessKind.READ}
        dtype = numpy.dtype(scalar_decl.dtype.name.lower())
        data.parameter_info[str(scalar_decl.name)] = ParameterInfo(access=access, dtype=dtype)

    data.unreferenced = [*sorted(name for name in accesses if accesses[name] == AccessKind.NONE)]
    _args_data_cache[pipeline.stencil_id] = data
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
            base_class=self.generate_base_class_name(),
            class_members=self.generate_class_members(),
            docstring=self.generate_docstring(),
            gt_backend=self.generate_backend_name(),
            gt_source=self.generate_sources(),
            gt_domain_info=repr(self.args_data.domain_info),
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
        return "from gt4py.stencil_object import StencilObject"

    def generate_class_name(self) -> str:
        """
        Generate the name of the stencil class.

        This should ususally be deferred to the chosen caching strategy via
        the builder object (see default implementation).
        """
        return self.builder.class_name

    def generate_base_class_name(self) -> str:
        return "StencilObject"

    def generate_docstring(self) -> str:
        """
        Generate the docstring of the stencil object.

        The default is to return the stencil definition's docstring or an
        empty string.
        The output should be least based on the stencil definition's docstring,
        if one exists.
        """
        return self.builder.gtir.docstring or ""

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
        if self.builder.gtir.sources is not None:
            return {
                key: gt_utils.text.format_source(value, line_length=self.SOURCE_LINE_LENGTH)
                for key, value in self.builder.gtir.sources.items()
            }
        return {}

    def generate_constants(self) -> Dict[str, str]:
        """
        Return a mapping of named numeric constants passed as externals.

        This is unlikely to require overriding.
        """
        if self.builder.gtir.externals:
            return {
                name: repr(value)
                for name, value in self.builder.gtir.externals.items()
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
        for arg in self.builder.gtir.api_signature:
            if arg.is_keyword:
                if arg.default:
                    keyword_args.append(
                        "{name}={default}".format(name=arg.name, default=arg.default)
                    )
                else:
                    keyword_args.append(arg.name)
            else:
                if arg.default:
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
