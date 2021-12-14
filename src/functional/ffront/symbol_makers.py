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

import ast
import builtins
import inspect
import types
import typing
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from eve import NodeTranslator
from eve.type_definitions import SourceLocation

from functional import common
from functional.common import GTSyntaxError, GTTypeError
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


def make_symbol_type(value: Any) -> foast.SymbolType:
    if isinstance(value, (type, types.GenericAlias)) or type(value).__module__ == "typing":
        return make_symbol_type_from_typing(value)
    elif isinstance(value, ast.AST):
        return make_symbol_type_from_AST_annotation(value)
    else:
        return make_symbol_type_from_value(value)


def make_symbol_type_from_typing(value: type) -> foast.SymbolType:
    if isinstance(value, type) and issubclass(value, (bool, int, float, np.generic)):
        return foast.ScalarType(kind=make_scalar_kind(value))
    elif isinstance(value, types.GenericAlias) or type(value).__module__ == "typing":
        origin = typing.get_origin(value)
        if origin == tuple:
            return foast.TupleType(
                types=[make_symbol_type_from_typing(arg) for arg in typing.get_args(value)]
            )
        elif origin == common.Field:
            args = typing.get_args(value)
            if len(args) != 2:
                raise common.GTTypeError(f"Field type requires two arguments, got {len(args)}!")

            dim_arg, dtype_arg = args
            dims: Union[Ellipsis, list[foast.Dimension]] = Ellipsis  # type: ignore[valid-type]

            if isinstance(dim_arg, list):
                for d in dim_arg:
                    if isinstance(d, common.Dimension):
                        dims.append(foast.Dimension(name=d.name))
                    else:
                        raise common.GTTypeError(f"Invalid field dimension definition '{d}'")

            elif dim_arg is Ellipsis:
                dims = dim_arg
            else:
                raise common.GTTypeError(f"Invalid field type dimensions '{dim_arg}'")

            dtype = make_symbol_type_from_typing(dtype_arg)
            if not isinstance(dtype, foast.ScalarType):
                raise common.GTTypeError(
                    "Field type dtype argument must be a scalar type (got '{dtype}')!"
                )
            return foast.FieldType(dims=dims, dtype=dtype)

    raise GTTypeError(f"'{value}' type is not supported")


def make_symbol_type_from_AST_annotation(value: type) -> foast.SymbolType:
    return FieldOperatorTypeParser.apply(value)


def make_symbol_type_from_value(value: Any) -> foast.SymbolType:
    match value:
        case bool() | int() | float() | np.generic():
            return foast.ScalarType(kind=make_scalar_kind(type(value)))
        case tuple() as tuple_value:
            return foast.TupleType(types=[make_symbol_type_from_value(t) for t in tuple_value])
        case types.FunctionType():
            # TODO (egparedes): recover the function signature from FieldOperator when possible
            sig = inspect.signature(value)
            if sig.return_annotation is inspect.Signature.empty:
                raise GTTypeError(
                    f"Referenced function '{value}' does not contain proper type annotations"
                )
            returns = make_symbol_type_from_typing(sig.return_annotation)

            args = []
            kwargs = []
            for p in sig.parameters.values():
                if p.annotation is inspect.Signature.empty:
                    raise GTTypeError(
                        f"Referenced function '{value}' does not contain proper type annotations"
                    )

                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    args.append(make_symbol_type_from_typing(p.annotation))
                elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwargs.append(make_symbol_type_from_typing(p.annotation))

            return foast.FunctionType(args=args, kwargs=kwargs, returns=returns)
        case _:
            raise common.GTTypeError(f"Impossible to map '{value}' value to a SymbolType")


def make_symbol_from_value(
    name: str, value: Any, namespace: foast.Namespace, location: SourceLocation
) -> foast.Symbol:
    symbol_type = make_symbol_type_from_value(value)
    if isinstance(symbol_type, foast.DataType):
        return foast.DataSymbol(id=name, type=symbol_type, namespace=namespace, location=location)
    elif isinstance(symbol_type, foast.FunctionType):
        return foast.Function(
            id=name,
            type=symbol_type,
            namespace=namespace,
            params=[],
            returns=[],
            location=location,
        )
    else:
        raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


class FieldOperatorTypeError(GTSyntaxError):
    def __init__(
        self,
        msg="",
        *,
        lineno=0,
        offset=0,
        filename=None,
        end_lineno=None,
        end_offset=None,
        text=None,
    ):
        msg = "Invalid Type Declaration: " + msg
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))


def _make_type_error(node, msg) -> FieldOperatorTypeError:
    return FieldOperatorTypeError(
        msg,
        lineno=getattr(node, "lineno", None),
        offset=getattr(node, "col_offset", None),
        end_lineno=getattr(node, "end_lineno", None),
        end_offset=getattr(node, "end_offset", None),
    )


class FieldOperatorTypeParser(NodeTranslator):
    """Parse type annotations into FOAST types.

    It expects to receive the AST node of an annotation.

    Examples
    --------
    >>> import ast
    >>>
    >>> test1 = ast.parse("test1: Field[..., float64]").body[0]
    >>> FieldOperatorTypeParser.apply(test1.annotation)  # doctest: +ELLIPSIS
    FieldType(dims=Ellipsis, dtype=ScalarType(kind=<ScalarKind.FLOAT64: ...>, shape=None))

    >>> test2 = ast.parse("test2: Field[['Foo'], 'int32']").body[0]
    >>> FieldOperatorTypeParser.apply(test2.annotation)  # doctest: +ELLIPSIS
    FieldType(dims=[Dimension(name='Foo')], dtype=ScalarType(kind=<ScalarKind.INT32: ...>, shape=None))

    >>> test3 = ast.parse("test3: Field['foo', bool]").body[0]
    >>> try:
    ...     FieldOperatorTypeParser.apply(test3.annotation)
    ... except FieldOperatorTypeError as err:
    ...     print(err.msg)
    Invalid Type Declaration: Field type dimension argument must be list or `...`!

    >>> test4 = ast.parse("test4: int").body[0]
    >>> FieldOperatorTypeParser.apply(test4.annotation)  # doctest: +ELLIPSIS
    ScalarType(kind=<ScalarKind.INT32: ...>, shape=None)

    >>> test5 = ast.parse("test5: tuple[Field[[X, Y, Z], float32], Field[[U, V], int64]]").body[0]
    >>> FieldOperatorTypeParser.apply(test5.annotation)  # doctest: +ELLIPSIS
    TupleType(types=[FieldType(...), FieldType(...)])
    """

    @classmethod
    def apply(cls, node: ast.AST) -> foast.SymbolType:
        return cls().visit(node)

    def visit_Subscript(self, node: ast.Subscript, **kwargs) -> foast.SymbolType:
        return self.visit(node.value, argument=node.slice, **kwargs)

    def visit_Name(
        self, node: ast.Name, *, argument: Optional[ast.AST] = None, **kwargs
    ) -> Union[foast.SymbolType, str]:
        maker = getattr(self, f"make_{node.id}", None)
        if maker is None:
            # TODO (ricoh): pull in type from external name
            return node.id
        return maker(argument)

    def visit_Constant(self, node: ast.Constant, **kwargs) -> Any:
        if isinstance(node.value, str) and (maker := getattr(self, f"make_{node.value}", None)):
            return maker(argument=None)
        else:
            return node.value

    def make_Field(self, argument: ast.Tuple) -> foast.FieldType:
        if not isinstance(argument, ast.Tuple) or len(argument.elts) != 2:
            nargs = len(getattr(argument, "elts", []))
            raise _make_type_error(argument, f"Field type requires two arguments, got {nargs}!")

        dim_arg, dtype_arg = argument.elts

        dims: Union[Ellipsis, list[foast.Dimension]] = Ellipsis  # type: ignore[valid-type]

        match dim_arg:
            case ast.Tuple() | ast.List():
                dims = [foast.Dimension(name=self.visit(dim)) for dim in argument.elts[0].elts]
            case ast.Ellipsis():
                dims = Ellipsis
            case _:
                dims = self.visit(dim_arg)
                if dims is not Ellipsis:
                    raise _make_type_error(
                        argument.elts[0], "Field type dimension argument must be list or `...`!"
                    )

        dtype = self.visit(dtype_arg)
        if not isinstance(dtype, foast.ScalarType):
            raise _make_type_error(dtype_arg, "Field type dtype argument must be a scalar type!")
        return foast.FieldType(dims=dims, dtype=dtype)

    def make_Tuple(self, argument: ast.Tuple) -> foast.TupleType:
        return foast.TupleType(types=[self.visit(element) for element in argument.elts])

    make_tuple = make_Tuple

    def make_int32(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=make_scalar_kind("int32"))

    def make_int64(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=make_scalar_kind("int64"))

    def make_int(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=make_scalar_kind("int"))

    def make_bool(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=make_scalar_kind("bool"))

    def make_float32(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=make_scalar_kind("float32"))

    def make_float64(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=make_scalar_kind("float64"))

    def make_float(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=make_scalar_kind("float"))
