import os
import typing
from typing import Any, Generic, Sequence, Tuple, TypeVar, Type
import functional.backend.defs as defs
import functional.backend.cpp as cpp

import jinja2

import eve.codegen
from eve import Node
from eve.codegen import JinjaTemplate, TemplatedGenerator


T = TypeVar("T")


class CommaSeparatedList(Node):
    items: list


class NewlineSeparatedList(Node):
    items: list


class BindingFunction(Node):
    exported_name: str
    wrapper_name: str
    doc: str


class BindingModule(Node):
    name: str
    doc: str
    functions: NewlineSeparatedList


class FunctionParameter(Node):
    name: str
    ndim: int
    dtype: Type


class FunctionCall(Node):
    target: defs.Function


class ReturnStmt(Node):
    expr: Any


class GridToolsSidExpr(Node):
    name: str
    num_dims: int
    dtype: typing.Type
    dim_config: int


class SidExpr(Node):
    identifier: str


class WrapperFunction(Node):
    name: str
    parameters: CommaSeparatedList
    body: Any


class BindingFile(Node):
    callee_header_file: str
    header_files: list[str]
    wrapper: WrapperFunction
    binding_module: BindingModule


class BindingCodeGenerator(TemplatedGenerator):
    BindingFile = JinjaTemplate(
        """\
        #include "{{callee_header_file}}"

        {% for header_file in header_files: %}\
        #include <{{header_file}}>
        {% endfor %}

        {{wrapper}}
        
        {{binding_module}}\
        """
    )

    WrapperFunction = JinjaTemplate(
        """\
        decltype(auto) {{name}}(
            {{parameters}}
        )
        {
            {{body}}
        }\
        """
    )

    def visit_FunctionParameter(self, param: FunctionParameter):
        if param.ndim > 0:
            type_str = "pybind11::buffer"
        else:
            type_str = cpp.render_python_type(param.dtype)
        return type_str + " " + param.name

    FunctionArgument = JinjaTemplate("""{{expr}}""")

    ReturnStmt = JinjaTemplate("""return {{expr}};""")

    def visit_GridToolsSidExpr(self, expr: GridToolsSidExpr):
        template = jinja2.Template(
            """\
            gridtools::as_sid<{{dtype}},\
                              {{num_dims}},\
                              gridtools::integral_constant<int, {{dim_config}}>,\
                              999'999'999>({{name}})\
            """
        )
        return template.render(
            name=expr.name,
            num_dims=expr.num_dims,
            dtype=cpp.render_python_type(expr.dtype),
            dim_config=expr.dim_config,
        )

    BindingModule = JinjaTemplate("""\
        PYBIND11_MODULE({{name}}, module) {
            module.doc() = "{{doc}}";
            {{functions}}
        }\
        """
    )

    BindingFunction = JinjaTemplate(
        """module.def("{{exported_name}}", &{{wrapper_name}}, "{{doc}}");"""
    )

    def visit_CommaSeparatedList(self, items: NewlineSeparatedList):
        str_items = [self.visit(item) for item in items.items]
        return ", ".join(str_items)

    def visit_NewlineSeparatedList(self, items: NewlineSeparatedList):
        str_items = [self.visit(item) for item in items.items]
        return str(os.linesep).join(str_items)

    def visit_FunctionCall(self, call: FunctionCall):
        args = [render_argument(index, param) for index, param in enumerate(call.target.parameters)]
        return cpp.render_function_call(call.target, args)


def make_parameter_list(parameters: Sequence[defs.ScalarParameter | defs.BufferParameter]) -> Sequence[FunctionParameter]:
    def make_parameter(parameter: [defs.ScalarParameter | defs.BufferParameter]):
        if isinstance(parameter, defs.ScalarParameter):
            return FunctionParameter(name=parameter.name, ndim=0, dtype=parameter.type_)
        else:
            return FunctionParameter(name=parameter.name, ndim=parameter.num_dimensions, dtype=parameter.scalar_type)
    regulars = [make_parameter(param) for param in parameters]
    return regulars


def render_argument(index: int, param: defs.ScalarParameter | defs.BufferParameter) -> str:
    if isinstance(param, defs.ScalarParameter):
        return """{name}""".format(name=param.name)
    else:
        template = jinja2.Template(
            """\
            gridtools::as_sid<{{dtype}},\
                              {{ndims}},\
                              gridtools::integral_constant<int, {{dim_config}}>,\
                              999'999'999>({{name}})\
            """
        )
        return template.render(
            name=param.name,
            ndims=param.num_dimensions,
            dtype=cpp.render_python_type(param.scalar_type),
            dim_config=index,
        )


def create_bindings(target: defs.Function, target_header: str) -> defs.BindingCodeModule:
    wrapper_name = target.name + "_wrapper"

    file_binding = BindingFile(
        callee_header_file=target_header,
        header_files=[
            "pybind11/pybind11.h",
            "pybind11/stl.h",
            "gridtools/storage/adapter/python_sid_adapter.hpp",
            "gridtools/common/defs.hpp",
            "gridtools/fn/unstructured.hpp",
            "gridtools/fn/cartesian.hpp",
            "gridtools/fn/backend/naive.hpp",
        ],
        wrapper=WrapperFunction(
            name=wrapper_name,
            parameters=CommaSeparatedList(items=make_parameter_list(target.parameters)),
            body=ReturnStmt(expr=FunctionCall(target=target))
        ),
        binding_module=BindingModule(
            name=target.name,
            doc="",
            functions=NewlineSeparatedList(
                items=[
                    BindingFunction(exported_name=target.name, wrapper_name=wrapper_name, doc="")
                ]
            ),
        ),
    )

    src = eve.codegen.format_source("cpp", BindingCodeGenerator.apply(file_binding), style="LLVM")
    return defs.BindingCodeModule(src, [defs.LibraryDependency("pybind11", "2.9.2")])
