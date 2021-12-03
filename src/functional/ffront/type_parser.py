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
from typing import Any, Optional, Union

from eve import NodeTranslator
from functional.common import GTSyntaxError
from functional.ffront import field_operator_ast as foast


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


def _get_field_args(node: ast.Tuple) -> dict:
    def _stringify(node) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        else:
            raise _make_type_error(node, "name or string required!")

    def _get_dimensions(
        node,
    ):
        if isinstance(node, ast.Ellipsis):
            return Ellipsis
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [foast.Dimension(name=_stringify(elt)) for elt in node.elts]
        else:
            raise _make_type_error(
                node, "Field dimension type argument must be a list of dimensions or `...`!"
            )

    return _get_dimensions(node.elts[0])


class FieldOperatorTypeParser(NodeTranslator):
    """
    Parse type annotations into FOAST types.

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
        if isinstance(node.value, str):
            maker = getattr(self, f"make_{node.value}", None)
            if maker is None:
                return node.value
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

    def make_tuple(self, argument: ast.Tuple) -> foast.TupleType:
        return self.make_Tuple(argument)

    def make_int32(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=foast.ScalarKind.INT32)

    def make_int64(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=foast.ScalarKind.INT64)

    def make_int(self, argument: None = None) -> foast.ScalarType:
        return self.make_int32()

    def make_bool(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=foast.ScalarKind.BOOL)

    def make_float32(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=foast.ScalarKind.FLOAT32)

    def make_float64(self, argument: None = None) -> foast.ScalarType:
        return foast.ScalarType(kind=foast.ScalarKind.FLOAT64)

    def make_float(self, argument: None = None) -> foast.ScalarType:
        return self.make_float64()
