# GT4Py Project - GridTools Framework
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

import builtins
import collections.abc
import functools
import types
import typing
from typing import Any, ForwardRef, Optional, Union

import numpy as np
import numpy.typing as npt

from eve import typingx
from eve.type_definitions import SourceLocation
from functional import common
from functional.ffront import field_operator_ast as foast


def make_scalar_kind(dtype: npt.DTypeLike) -> foast.ScalarKind:
    try:
        dt = np.dtype(dtype)
    except TypeError as err:
        raise common.GTTypeError(f"Invalid scalar type definition ({dtype})") from err

    if dt.shape == () and dt.fields is None:
        match dt:
            case np.bool_:
                return foast.ScalarKind.BOOL
            case np.int32:
                return foast.ScalarKind.INT32
            case np.int64:
                return foast.ScalarKind.INT64
            case np.float32:
                return foast.ScalarKind.FLOAT32
            case np.float64:
                return foast.ScalarKind.FLOAT64
            case _:
                raise common.GTTypeError(f"Impossible to map '{dtype}' value to a ScalarKind")
    else:
        raise common.GTTypeError(f"Non-trivial dtypes like '{dtype}' are not yet supported")


def make_symbol_type_from_typing(
    value: Any,
    *,
    global_ns: Optional[dict[str, Any]] = None,
    local_ns: Optional[dict[str, Any]] = None,
) -> foast.SymbolType:
    recursive_make_symbol = functools.partial(
        make_symbol_type_from_typing, global_ns=global_ns, local_ns=local_ns
    )

    # ForwardRef
    if isinstance(value, str):
        value = ForwardRef(value)
    if type(value).__module__ == "typing" and hasattr(value, "__forward_arg__"):
        try:
            value = typingx.resolve_type(
                value, global_ns=global_ns, local_ns=local_ns, allow_partial=False
            )
        except Exception as error:
            raise FieldOperatorTypeError(
                f"Type annotation ({value}) has undefined forward references!"
            ) from error

    # Annotated
    if typing.get_origin(value) is typing.Annotated:
        value, extra_args = typing.get_args(value)
        while typing.get_origin(value) is typing.Annotated:
            value = typing.get_origin(value)
    else:
        extra_args = None

    value_type = (
        typing.get_origin(value)
        if isinstance(value, types.GenericAlias) or type(value).__module__ == "typing"
        else value
    )
    args = typing.get_args(value)

    match value_type:
        case type() as t if issubclass(t, (bool, int, float, np.generic)):
            return foast.ScalarType(kind=make_scalar_kind(value))

        case builtins.tuple:
            if not args:
                raise FieldOperatorTypeError(
                    f"Tuple annotation ({value}) requires at least one argument!"
                )
            return foast.TupleType(types=[recursive_make_symbol(arg) for arg in args])

        case common.Field:
            if len(args) != 2:
                raise FieldOperatorTypeError(f"Field type requires two arguments, got {len(args)}!")

            dims: Union[Ellipsis, list[foast.Dimension]] = []
            dim_arg, dtype_arg = args
            if isinstance(dim_arg, list):
                for d in dim_arg:
                    if not isinstance(d, common.Dimension):
                        raise FieldOperatorTypeError(f"Invalid field dimension definition '{d}'")
                    dims.append(foast.Dimension(name=d.value))
            elif dim_arg is Ellipsis:
                dims = dim_arg
            else:
                raise FieldOperatorTypeError(f"Invalid field type dimensions '{dim_arg}'")

            dtype = recursive_make_symbol(dtype_arg)
            if not isinstance(dtype, foast.ScalarType):
                raise FieldOperatorTypeError(
                    "Field type dtype argument must be a scalar type (got '{dtype}')!"
                )
            return foast.FieldType(dims=dims, dtype=dtype)

        case collections.abc.Callable:
            arg_types, return_type = args
            if arg_types in (None, Ellipsis) or return_type is None:
                raise FieldOperatorTypeError("Not annotated functions are not supported!")

            if not isinstance(extra_args, typingx.CallableKwargsInfo):
                raise FieldOperatorTypeError(f"Invalid callable annotations in {value}")

            return foast.FunctionType(
                args=[recursive_make_symbol(arg) for arg in arg_types],
                kwargs={
                    arg: recursive_make_symbol(arg_type)
                    for arg, arg_type in extra_args.data.items()
                },
                returns=recursive_make_symbol(return_type),
            )

    raise FieldOperatorTypeError(f"'{value}' type is not supported")


def make_symbol_from_value(
    name: str, value: Any, namespace: foast.Namespace, location: SourceLocation
) -> foast.Symbol:
    if not isinstance(value, type) or type(value).__module__ != "typing":
        value = typingx.get_typing(value, annotate_callable_kwargs=True)
    symbol_type = make_symbol_type_from_typing(value)
    if isinstance(symbol_type, foast.DataType):
        return foast.DataSymbol(id=name, type=symbol_type, namespace=namespace, location=location)
    elif isinstance(symbol_type, foast.FunctionType):
        return foast.Function(
            id=name,
            type=symbol_type,
            namespace=namespace,
            location=location,
        )
    else:
        raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


class FieldOperatorTypeError(common.GTTypeError):
    def __init__(
        self,
        msg="",
        *,
        info=None,
    ):
        msg = f"Invalid type declaration: {msg}"
        args = tuple([msg, info] if info else [msg])
        super().__init__(*args)


def _make_type_error(node, msg) -> FieldOperatorTypeError:
    return FieldOperatorTypeError(
        msg,
        lineno=getattr(node, "lineno", None),
        offset=getattr(node, "col_offset", None),
        end_lineno=getattr(node, "end_lineno", None),
        end_offset=getattr(node, "end_offset", None),
    )


# class FieldOperatorTypeParser(NodeTranslator):
#     """Parse type annotations into FOAST types.

#     It expects to receive the AST node of an annotation.

#     Examples
#     --------
#     >>> import ast
#     >>>
#     >>> test1 = ast.parse("test1: Field[..., float64]").body[0]
#     >>> FieldOperatorTypeParser.apply(test1.annotation)  # doctest: +ELLIPSIS
#     FieldType(dims=Ellipsis, dtype=ScalarType(kind=<ScalarKind.FLOAT64: ...>, shape=None))

#     >>> test2 = ast.parse("test2: Field[['Foo'], 'int32']").body[0]
#     >>> FieldOperatorTypeParser.apply(test2.annotation)  # doctest: +ELLIPSIS
#     FieldType(dims=[Dimension(name='Foo')], dtype=ScalarType(kind=<ScalarKind.INT32: ...>, shape=None))

#     >>> test3 = ast.parse("test3: Field['foo', bool]").body[0]
#     >>> try:
#     ...     FieldOperatorTypeParser.apply(test3.annotation)
#     ... except FieldOperatorTypeError as err:
#     ...     print(err.msg)
#     Invalid Type Declaration: Field type dimension argument must be list or `...`!

#     >>> test4 = ast.parse("test4: int").body[0]
#     >>> FieldOperatorTypeParser.apply(test4.annotation)  # doctest: +ELLIPSIS
#     ScalarType(kind=<ScalarKind.INT32: ...>, shape=None)

#     >>> test5 = ast.parse("test5: tuple[Field[[X, Y, Z], float32], Field[[U, V], int64]]").body[0]
#     >>> FieldOperatorTypeParser.apply(test5.annotation)  # doctest: +ELLIPSIS
#     TupleType(types=[FieldType(...), FieldType(...)])
#     """

#     @classmethod
#     def apply(cls, node: ast.AST) -> foast.SymbolType:
#         return cls().visit(node)

#     def visit_Subscript(self, node: ast.Subscript, **kwargs) -> foast.SymbolType:
#         return self.visit(node.value, argument=node.slice, **kwargs)

#     def visit_Name(
#         self, node: ast.Name, *, argument: Optional[ast.AST] = None, **kwargs
#     ) -> Union[foast.SymbolType, str]:
#         maker = getattr(self, f"make_{node.id}", None)
#         if maker is None:
#             # TODO (ricoh): pull in type from external name
#             return node.id
#         return maker(argument)

#     def visit_Constant(self, node: ast.Constant, **kwargs) -> Any:
#         if isinstance(node.value, str) and (maker := getattr(self, f"make_{node.value}", None)):
#             return maker(argument=None)
#         else:
#             return node.value

#     def make_Field(self, argument: ast.Tuple) -> foast.FieldType:
#         if not isinstance(argument, ast.Tuple) or len(argument.elts) != 2:
#             nargs = len(getattr(argument, "elts", []))
#             raise _make_type_error(argument, f"Field type requires two arguments, got {nargs}!")

#         dim_arg, dtype_arg = argument.elts

#         dims: Union[Ellipsis, list[foast.Dimension]] = Ellipsis  # type: ignore[valid-type]

#         match dim_arg:
#             case ast.Tuple() | ast.List():
#                 dims = [foast.Dimension(name=self.visit(dim)) for dim in argument.elts[0].elts]
#             case ast.Ellipsis():
#                 dims = Ellipsis
#             case _:
#                 dims = self.visit(dim_arg)
#                 if dims is not Ellipsis:
#                     raise _make_type_error(
#                         argument.elts[0], "Field type dimension argument must be list or `...`!"
#                     )

#         dtype = self.visit(dtype_arg)
#         if not isinstance(dtype, foast.ScalarType):
#             raise _make_type_error(dtype_arg, "Field type dtype argument must be a scalar type!")
#         return foast.FieldType(dims=dims, dtype=dtype)

#     def make_Tuple(self, argument: ast.Tuple) -> foast.TupleType:
#         return foast.TupleType(types=[self.visit(element) for element in argument.elts])

#     make_tuple = make_Tuple

#     def make_int32(self, argument: None = None) -> foast.ScalarType:
#         return foast.ScalarType(kind=make_scalar_kind("int32"))

#     def make_int64(self, argument: None = None) -> foast.ScalarType:
#         return foast.ScalarType(kind=make_scalar_kind("int64"))

#     def make_int(self, argument: None = None) -> foast.ScalarType:
#         return foast.ScalarType(kind=make_scalar_kind("int"))

#     def make_bool(self, argument: None = None) -> foast.ScalarType:
#         return foast.ScalarType(kind=make_scalar_kind("bool"))

#     def make_float32(self, argument: None = None) -> foast.ScalarType:
#         return foast.ScalarType(kind=make_scalar_kind("float32"))

#     def make_float64(self, argument: None = None) -> foast.ScalarType:
#         return foast.ScalarType(kind=make_scalar_kind("float64"))

#     def make_float(self, argument: None = None) -> foast.ScalarType:
#         return foast.ScalarType(kind=make_scalar_kind("float"))
