# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Python bindings generator for C++ functions."""

from __future__ import annotations

from typing import Any, Optional, Sequence, TypeVar, Union

import gt4py.eve as eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from gt4py.next import common, config
from gt4py.next.otf import cpp_utils, languages, stages, workflow
from gt4py.next.otf.binding import cpp_interface, interface
from gt4py.next.type_system import type_specifications as ts


SrcL = TypeVar("SrcL", bound=languages.NanobindSrcL, covariant=True)


class Expr(eve.Node):
    pass


class DimensionSpec(Expr):
    name: str
    static_stride: Optional[int]


class BufferSID(Expr):
    source_buffer: str
    dimensions: Sequence[DimensionSpec]
    scalar_type: ts.ScalarType


class Tuple(Expr):
    elems: list[Union[Expr, str]]


class FunctionCall(Expr):
    target: interface.Function
    args: Sequence[Any]


class ReturnStmt(eve.Node):
    expr: Expr


class FunctionParameter(eve.Node):
    name: str
    type_: ts.TypeSpec


class WrapperFunction(eve.Node):
    name: str
    parameters: Sequence[FunctionParameter]
    body: ReturnStmt


class BindingFunction(eve.Node):
    exported_name: str
    wrapper_name: str
    doc: str


class BindingModule(eve.Node):
    name: str
    doc: str
    functions: Sequence[BindingFunction]


class BindingFile(eve.Node):
    callee_header_file: str
    header_files: Sequence[str]
    wrapper: WrapperFunction
    binding_module: BindingModule


def _type_string(type_: ts.TypeSpec) -> str:
    if isinstance(type_, ts.TupleType):
        return f"std::tuple<{','.join(_type_string(t) for t in type_.types)}>"
    elif isinstance(type_, ts.FieldType):
        ndims = len(type_.dims)
        # cannot be ListType: the concept is represented as Field with local Dimension in this interface
        assert isinstance(type_.dtype, ts.ScalarType)
        dtype = cpp_utils.pytype_to_cpptype(type_.dtype)
        shape = f"nanobind::shape<{', '.join(['gridtools::nanobind::dynamic_size'] * ndims)}>"
        buffer_t = f"nanobind::ndarray<{dtype}, {shape}>"
        origin_t = f"std::tuple<{', '.join(['ptrdiff_t'] * ndims)}>"
        return f"std::pair<{buffer_t}, {origin_t}>"
    elif isinstance(type_, ts.ScalarType):
        return cpp_utils.pytype_to_cpptype(type_)
    else:
        raise ValueError(f"Type '{type_}' is not supported in nanobind interfaces.")


class BindingCodeGenerator(TemplatedGenerator):
    BindingFile = as_jinja(
        """\
        #include "{{callee_header_file}}"

        {% for header_file in header_files: %}\
        #include <{{header_file}}>
        {% endfor %}

        {{wrapper}}

        {{binding_module}}\
        """
    )

    WrapperFunction = as_jinja(
        """\
        decltype(auto) {{name}}(
            {{"\n,".join(parameters)}}
        )
        {
            {{body}}
        }\
        """
    )

    FunctionParameter = as_jinja("{{_this_module._type_string(_this_node.type_)}} {{name}}")

    ReturnStmt = as_jinja("""return {{expr}};""")

    BindingModule = as_jinja(
        """\
        NB_MODULE({{name}}, module) {
            module.doc() = "{{doc}}";
            {{"\n".join(functions)}}
        }\
        """
    )

    BindingFunction = as_jinja("""module.def("{{exported_name}}", &{{wrapper_name}}, "{{doc}}");""")

    def visit_FunctionCall(self, call: FunctionCall) -> str:
        args = [self.visit(arg) for arg in call.args]
        return cpp_interface.render_function_call(call.target, args)

    def visit_BufferSID(self, sid: BufferSID, **kwargs: Any) -> str:
        pybuffer = f"{sid.source_buffer}.first"
        dims = [self.visit(dim) for dim in sid.dimensions]
        origin = f"{sid.source_buffer}.second"
        stride_spec = [
            "gridtools::nanobind::dynamic_size"
            if dim.static_stride is None
            else str(dim.static_stride)
            for dim in sid.dimensions
        ]
        stride_spec_string = (
            f"gridtools::nanobind::stride_spec<{', '.join(str(s) for s in stride_spec)}>{{}}"
        )

        as_sid = f"gridtools::nanobind::as_sid({pybuffer}, {stride_spec_string})"
        shifted = f"gridtools::sid::shift_sid_origin({as_sid}, {origin})"
        renamed = f"gridtools::sid::rename_numbered_dimensions<{', '.join(dims)}>({shifted})"
        return renamed

    Tuple = as_jinja("""gridtools::tuple({{','.join(elems)}})""")

    DimensionSpec = as_jinja("""generated::{{name}}_t""")


def _tuple_get(index: int, var: str) -> str:
    return f"gridtools::tuple_util::get<{index}>({var})"


def make_argument(name: str, type_: ts.TypeSpec) -> str | BufferSID | Tuple:
    if isinstance(type_, ts.FieldType):
        return BufferSID(
            source_buffer=name,
            dimensions=[
                DimensionSpec(
                    name=dim.value,
                    static_stride=1
                    if (
                        config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                        and dim.kind == common.DimensionKind.HORIZONTAL
                    )
                    else None,
                )
                for dim in type_.dims
            ],
            scalar_type=type_.dtype,
        )
    elif isinstance(type_, ts.TupleType):
        elements = [make_argument(_tuple_get(i, name), t) for i, t in enumerate(type_.types)]
        return Tuple(elems=elements)
    elif isinstance(type_, ts.ScalarType):
        return name
    else:
        raise ValueError(f"Type '{type_}' is not supported in nanobind interfaces.")


def create_bindings(
    program_source: stages.ProgramSource[SrcL, languages.LanguageWithHeaderFilesSettings],
) -> stages.BindingSource[SrcL, languages.Python]:
    """
    Generate Python bindings through which a C++ function can be called.

    Parameters
    ----------
    program_source
        The program source for which the bindings are created
    """
    if program_source.language not in [languages.CPP, languages.CUDA, languages.HIP]:
        raise ValueError(
            f"Can only create bindings for C++ program sources, received '{program_source.language}'."
        )
    wrapper_name = program_source.entry_point.name + "_wrapper"

    file_binding = BindingFile(
        callee_header_file=f"{program_source.entry_point.name}.{program_source.language_settings.header_extension}",
        header_files=[
            "nanobind/nanobind.h",
            "nanobind/stl/tuple.h",
            "nanobind/stl/pair.h",
            "nanobind/ndarray.h",
            "gridtools/sid/composite.hpp",
            "gridtools/sid/unknown_kind.hpp",
            "gridtools/sid/rename_dimensions.hpp",
            "gridtools/common/defs.hpp",
            "gridtools/common/tuple_util.hpp",
            "gridtools/fn/unstructured.hpp",
            "gridtools/fn/cartesian.hpp",
            "gridtools/storage/adapter/nanobind_adapter.hpp",
        ],
        wrapper=WrapperFunction(
            name=wrapper_name,
            parameters=[
                FunctionParameter(name=param.name, type_=param.type_)
                for param in program_source.entry_point.parameters
            ],
            body=ReturnStmt(
                expr=FunctionCall(
                    target=program_source.entry_point,
                    args=[
                        make_argument(param.name, param.type_)
                        for param in program_source.entry_point.parameters
                    ],
                )
            ),
        ),
        binding_module=BindingModule(
            name=program_source.entry_point.name,
            doc="",
            functions=[
                BindingFunction(
                    exported_name=program_source.entry_point.name, wrapper_name=wrapper_name, doc=""
                )
            ],
        ),
    )

    src = interface.format_source(
        program_source.language_settings, BindingCodeGenerator.apply(file_binding)
    )

    return stages.BindingSource(src, (interface.LibraryDependency("nanobind", "2.0.0"),))


@workflow.make_step
def bind_source(
    inp: stages.ProgramSource[SrcL, languages.LanguageWithHeaderFilesSettings],
) -> stages.CompilableSource[SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python]:
    return stages.CompilableSource(program_source=inp, binding_source=create_bindings(inp))
