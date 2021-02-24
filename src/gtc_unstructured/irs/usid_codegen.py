# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

from types import MappingProxyType
from typing import ClassVar, Mapping

from devtools import debug  # noqa: F401

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gtc_unstructured.irs import common
from gtc_unstructured.irs.usid import (
    Computation,
    Connectivity,
    Kernel,
    KernelCall,
    NeighborLoop,
    SidCompositeSparseEntry,
    Temporary,
    TemporarySparseField,
)


class UsidCodeGenerator(codegen.TemplatedGenerator):
    DATA_TYPE_TO_STR: ClassVar[Mapping[common.DataType, str]] = MappingProxyType(
        {
            common.DataType.BOOLEAN: "bool",
            common.DataType.INT32: "int",
            common.DataType.UINT32: "unsigned_int",
            common.DataType.FLOAT32: "float",
            common.DataType.FLOAT64: "double",
        }
    )

    BUILTIN_LITERAL_TO_STR: ClassVar[Mapping[common.BuiltInLiteral, str]] = MappingProxyType(
        {
            common.BuiltInLiteral.MAX_VALUE: "std::numeric_limits<double>::max()",  # TODO: datatype
            common.BuiltInLiteral.MIN_VALUE: "std::numeric_limits<double>::min()",
            common.BuiltInLiteral.ZERO: "0",
            common.BuiltInLiteral.ONE: "1",
        }
    )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def location_type_from_dimensions(self, dimensions):
        location_type = [dim for dim in dimensions if isinstance(dim, common.LocationType)]
        if len(location_type) != 1:
            raise ValueError("Doesn't contain a LocationType!")
        return location_type[0]

    headers_ = [
        "<gridtools/common/gt_math.hpp>",
        "<gridtools/common/array.hpp>",
        "<gridtools/usid/dim.hpp>",
        "<gridtools/usid/helpers.hpp>",
        "<gridtools/common/gt_math.hpp>",
    ]

    namespace_ = None

    preface_ = ""

    def visit_LocationType(self, node: common.LocationType, **kwargs):
        return {
            common.LocationType.Vertex: "vertex",
            common.LocationType.Edge: "edge",
            common.LocationType.Cell: "cell",
        }[node]

    def visit_bool(self, node: bool, **kwargs):
        if node:
            return "true"
        else:
            return "false"

    def visit_SidCompositeSparseEntry(self, node: SidCompositeSparseEntry, **kwargs):
        return self.generic_visit(
            node, connectivity_tag=kwargs["symtable"][node.connectivity].tag, **kwargs
        )

    SidCompositeSparseEntry = as_fmt("sid::rename_dimensions<dim::s, {connectivity_tag}>({ref})")

    SidCompositeEntry = as_fmt("{ref}")

    SidComposite = as_mako(
        """
        sid::composite::make<${ ','.join([t.name for t in _this_node.entries]) }>(
        ${ ','.join(entries)})
        """
    )

    def visit_KernelCall(self, node: KernelCall, **kwargs):
        kernel: Kernel = kwargs["symtable"][node.name]
        domain = f"d.{self.visit(kernel.primary_location)}"

        sids = self.visit([kernel.primary_composite] + kernel.secondary_composites, **kwargs)

        return self.generic_visit(node, domain=domain, sids=sids)

    KernelCall = as_mako(
        """
        call_kernel<${name}>(${domain}, d.k, ${','.join(sids)});
        """
    )

    FieldAccess = as_mako(
        """<%
            composite_deref = symtable[_this_node.sid]
            sid_entry_deref = symtable[_this_node.name]
        %>field<${ sid_entry_deref.name }>(${ composite_deref.ptr_name })"""
    )

    ArrayAccess = as_fmt("{name}[{subscript}]")

    AssignStmt = as_fmt("{left} = {right};")

    NativeFuncCall = as_fmt("gridtools::math::{func}({','.join(args)})")  # TODO: fix func

    BinaryOp = as_fmt("({left} {op} {right})")

    PtrRef = as_fmt("{name}")

    LocalIndex = as_fmt("{name}")

    def visit_NeighborLoop(self, node: NeighborLoop, symtable, **kwargs):
        primary_sid_deref = symtable[node.primary_sid]
        connectivity_deref = symtable[node.connectivity]
        indexed = ""
        index_var = ""
        if node.local_index:
            indexed = "_indexed"
            index_var = f", auto {self.visit(node.local_index)}"
        return self.generic_visit(
            node,
            symtable={
                **node.symtable_,
                **symtable,
            },  # should be partly bounded (should see only global scope (tags) and current scope)
            primary_sid_deref=primary_sid_deref,
            connectivity_deref=connectivity_deref,
            indexed=indexed,
            index_var=index_var,
            **kwargs,
        )

    # TODO consider stricter capture
    NeighborLoop = as_mako(
        """
        foreach_neighbor${indexed}<${connectivity_deref.tag}>([&](auto &&${primary}, auto &&${secondary}${index_var}){${''.join(body)}}, ${primary_sid_deref.ptr_name}, ${primary_sid_deref.strides_name}, ${secondary_sid});
        """
    )

    Literal = as_mako(
        """<%
            literal= _this_node.value if isinstance(_this_node.value, str) else _this_generator.BUILTIN_LITERAL_TO_STR[_this_node.value]
        %>(${ _this_generator.DATA_TYPE_TO_STR[_this_node.vtype] })${ literal }"""
    )

    VarAccess = as_fmt("{name}")

    VarDecl = as_mako(
        "${ _this_generator.DATA_TYPE_TO_STR[_this_node.vtype] } ${ name } = ${ init };"
    )

    StaticArrayDecl = as_mako(
        "gridtools::array<${_this_generator.DATA_TYPE_TO_STR[_this_node.vtype]}, ${size}> ${name} = {${','.join(init)}};"
    )

    def visit_Connectivity(self, node: Connectivity, **kwargs):
        c_has_skip_values = "true" if node.has_skip_values else "false"
        return self.generic_visit(node, c_has_skip_values=c_has_skip_values)

    Connectivity = as_mako(
        "struct ${_this_node.tag}: connectivity<${max_neighbors},${c_has_skip_values}>{};"
    )

    def visit_Temporary(self, node: Temporary, **kwargs):
        c_vtype = self.DATA_TYPE_TO_STR[node.vtype]
        loctype = self.visit(self.location_type_from_dimensions(node.dimensions))
        return self.generic_visit(node, loctype=loctype, c_vtype=c_vtype, **kwargs)

    Temporary = as_mako(
        """
        auto ${ name } = make_simple_tmp_storage<${ c_vtype }>(
            d.${ loctype }, d.k, alloc);"""
    )

    def visit_TemporarySparseField(self, node: TemporarySparseField, *, symtable, **kwargs):
        c_vtype = self.DATA_TYPE_TO_STR[node.vtype]
        loctype = self.visit(self.location_type_from_dimensions(node.dimensions))
        connectivity_deref = symtable[node.connectivity]
        return self.generic_visit(
            node,
            s_size=connectivity_deref.max_neighbors,
            c_vtype=c_vtype,
            loctype=loctype,
            **kwargs,
        )

    TemporarySparseField = as_mako(
        """
        auto ${ name } = make_simple_sparse_tmp_storage<${ c_vtype }>(
            d.${ loctype }, d.k, ${s_size}, alloc);"""
    )

    def visit_Kernel(self, node: Kernel, symtable, **kwargs):
        primary_signature = f"auto && {node.primary_composite.ptr_name}, auto&& {node.primary_composite.strides_name}"
        secondary_signature = (
            ""
            if len(node.secondary_composites) == 0
            else ", auto &&" + ", auto&&".join(c.name for c in node.secondary_composites)
        )
        return self.generic_visit(
            node,
            symtable={**symtable, **node.symtable_},
            primary_signature=primary_signature,
            secondary_signature=secondary_signature,
            **kwargs,
        )

    Kernel = as_mako(
        """
        struct ${name} {
            GT_FUNCTION auto operator()() const {
                return [](${primary_signature}${secondary_signature}){
                    ${''.join(body)}
                };
            }
        };
        """
    )

    def visit_Computation(self, node: Computation, **kwargs):
        # maybe tags should be generated in lowering
        field_tags = set()
        for field in node.parameters + node.temporaries:
            field_tags.add("struct " + field.tag + ";")

        connectivity_params = [f"auto&& {c.name}" for c in node.connectivities]
        field_params = [f"auto&& {f.name}" for f in node.parameters]

        connectivity_fields = [
            f"{c.name} = sid::rename_dimensions<dim::n, {c.tag}>(std::forward<decltype({c.name})>({c.name})(traits_t()))"
            for c in node.connectivities
        ]

        return self.generic_visit(
            node,
            field_tags=field_tags,
            connectivity_params=connectivity_params,
            connectivity_fields=connectivity_fields,
            field_params=field_params,
            symtable=node.symtable_,
            **kwargs,
        )

    Computation = as_mako(
        """
        ${ '\\n'.join('#include ' + header for header in _this_generator.headers_) }


        namespace ${ name }_impl_ {
            using namespace gridtools;
            using namespace gridtools::usid;
            using namespace gridtools::usid::${_this_generator.namespace_};
            ${ ''.join(connectivities)}
            ${ ''.join(field_tags) }

            ${ ''.join(kernels) }


            auto ${name} = [](domain d
                %if connectivity_params:
                , ${','.join(connectivity_params)}
                %endif
                ) {
                ${ ''.join(f"static_assert(is_sid<decltype({c.name}(traits_t()))>());" for c in _this_node.connectivities)}
                return
                    [d = std::move(d)
                    %if connectivity_fields:
                    , ${','.join(connectivity_fields)}
                    %endif
                            ](
                        ${','.join(field_params)}
                            ){
                            ${ ''.join(f"static_assert(is_sid<decltype({p.name})>());" for p in _this_node.parameters)}
                            %if temporaries:
                            auto alloc = make_allocator();
                            %endif
                            ${''.join(temporaries)}

                            ${''.join(ctrlflow_ast)}

                            };

            };
        }

        using ${ name }_impl_::${name};
        """
    )


class UsidGpuCodeGenerator(UsidCodeGenerator):

    headers_ = UsidCodeGenerator.headers_ + [
        "<gridtools/usid/cuda_helpers.hpp>",
    ]

    namespace_ = "cuda"

    preface_ = (
        UsidCodeGenerator.preface_
        + """
        #ifndef __CUDACC__
        #error "Tried to compile CUDA code with a regular C++ compiler."
        #endif
    """
    )


class UsidNaiveCodeGenerator(UsidCodeGenerator):

    headers_ = UsidCodeGenerator.headers_ + [
        "<gridtools/usid/naive_helpers.hpp>",
    ]

    namespace_ = "naive"
