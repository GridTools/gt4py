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

from __future__ import annotations

import ast
import copy
import functools
import inspect
import symtable
import textwrap
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Optional, Union

from eve import typingx
from eve.type_definitions import SourceLocation
from functional import common
from functional.ffront import common_types, fbuiltins
from functional.ffront import field_operator_ast as foast
from functional.ffront import symbol_makers
from functional.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    StringifyAnnotationsPass,
    UnpackedAssignPass,
)
from functional.ffront.foast_passes.shift_recognition import FieldOperatorShiftRecognition
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction


MISSING_FILENAME = "<string>"


def make_source_definition_from_function(func: Callable) -> SourceDefinition:
    try:
        filename = inspect.getabsfile(func) or MISSING_FILENAME
        source = textwrap.dedent(inspect.getsource(func))
        starting_line = (
            inspect.getsourcelines(func)[1] if not filename.endswith(MISSING_FILENAME) else 1
        )
    except OSError as err:
        if filename.endswith(MISSING_FILENAME):
            message = "Can not create field operator from a function that is not in a source file!"
        else:
            message = f"Can not get source code of passed function ({func})"
        raise ValueError(message) from err

    return SourceDefinition(source, filename, starting_line)


def make_closure_refs_from_function(func: Callable) -> ClosureRefs:
    (nonlocals, globals, inspect_builtins, inspect_unbound) = inspect.getclosurevars(  # noqa: A001
        func
    )
    # python builtins returned by getclosurevars() are not ffront.builtins
    unbound = set(inspect_builtins.keys()) | inspect_unbound
    builtins = unbound & set(fbuiltins.ALL_BUILTIN_NAMES)
    unbound -= builtins
    annotations = typingx.get_type_hints(func)

    return ClosureRefs(nonlocals, globals, annotations, builtins, unbound)


def make_symbol_names_from_source(source: str, filename: str = MISSING_FILENAME) -> SymbolNames:
    try:
        mod_st = symtable.symtable(source, filename, "exec")
    except SyntaxError as err:
        raise common.GTValueError(
            f"Unexpected error when parsing provided source code (\n{source}\n)"
        ) from err

    assert mod_st.get_type() == "module"
    if len(children := mod_st.get_children()) != 1:
        raise common.GTValueError(
            f"Sources with multiple function definitions are not yet supported (\n{source}\n)"
        )

    func_st = children[0]

    param_names = set()
    imported_names = set()
    local_names = set()
    for name in func_st.get_locals():
        if (s := func_st.lookup(name)).is_imported():
            imported_names.add(name)
        elif s.is_parameter:
            param_names.add(name)
        else:
            local_names.add(name)

    # symtable returns regular free (or non-local) variables in 'get_frees()' and
    # the free variables introduced with the 'nonlocal' statement in 'get_nonlocals()'
    nonlocal_names = set(func_st.get_frees()) | set(func_st.get_nonlocals())
    global_names = set(func_st.get_globals()) - set(fbuiltins.ALL_BUILTIN_NAMES)

    return SymbolNames(
        params=param_names,
        locals=local_names,
        imported=imported_names,
        nonlocals=nonlocal_names,
        globals=global_names,
    )


@dataclass(frozen=True)
class SourceDefinition:
    """
    A GT4Py source code definition encoded as a string.

    It can be created from an actual function object using :meth:`from_function()`.
    It also supports unpacking.


    Examples
    -------
    >>> def foo(a):
    ...     return a
    >>> src_def = SourceDefinition.from_function(foo)
    >>> print(src_def)
    SourceDefinition(source='def foo(a):... starting_line=1)

    >>> source, filename, starting_line = src_def
    >>> print(source)
    def foo(a):
        return a
    ...
    """

    source: str
    filename: str = MISSING_FILENAME
    starting_line: int = 1

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        yield from iter((self.source, self.filename, self.starting_line))

    from_function = staticmethod(make_source_definition_from_function)


@dataclass(frozen=True)
class ClosureRefs:
    """
    Mappings from names used in a Python function to the actual values.

    It can be created from an actual function object using :meth:`from_function()`.
    It also supports unpacking.
    """

    nonlocals: dict[str, Any]
    globals: dict[str, Any]  # noqa: A003  # shadowing a python builtin
    annotations: dict[str, Any]
    builtins: set[str]
    unbound: set[str]

    def __iter__(self) -> Iterator[Union[dict[str, Any], set[str]]]:
        yield from iter(
            (self.nonlocals, self.globals, self.annotations, self.builtins, self.unbound)
        )

    from_function = staticmethod(make_closure_refs_from_function)


@dataclass(frozen=True)
class SymbolNames:
    """
    Collection of symbol names used in a function classified by kind.

    It can be created directly from source code using :meth:`from_source()`.
    It also supports unpacking.
    """

    params: tuple[str, ...]
    locals: tuple[str, ...]  # noqa: A003  # shadowing a python builtin
    imported: tuple[str, ...]
    nonlocals: tuple[str, ...]
    globals: tuple[str, ...]  # noqa: A003  # shadowing a python builtin

    @functools.cached_property
    def all_locals(self) -> tuple[str]:
        return self.params + self.locals + self.imported

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        yield from iter((self.params, self.locals, self.imported, self.nonlocals, self.globals))

    from_source = staticmethod(make_symbol_names_from_source)


@dataclass(frozen=True, kw_only=True)
class FieldOperatorParser(ast.NodeVisitor):
    """
    Parse field operator function definition from source code into FOAST.

    Catch any Field Operator specific syntax errors and typing problems.

    Example
    -------
    Parse a function into a Field Operator AST (FOAST), which can
    be lowered into Iterator IR (ITIR)

    >>> from functional.common import Field
    >>> float64 = float
    >>> def field_op(inp: Field[..., float64]):
    ...     return inp
    >>> foast_tree = FieldOperatorParser.apply_to_function(field_op)
    >>> foast_tree  # doctest: +ELLIPSIS
    FieldOperator(..., id='field_op', ...)
    >>> foast_tree.params  # doctest: +ELLIPSIS
    [FieldSymbol(..., id='inp', ...)]
    >>> foast_tree.body  # doctest: +ELLIPSIS
    [Return(..., value=Name(..., id='inp'))]


    If a syntax error is encountered, it will point to the location in the source code.

    >>> def wrong_syntax(inp: Field[..., int]):
    ...     for i in [1, 2, 3]: # for is not part of the field operator syntax
    ...         tmp = inp
    ...     return tmp
    >>>
    >>> try:
    ...     FieldOperatorParser.apply_to_function(wrong_syntax)
    ... except FieldOperatorSyntaxError as err:
    ...     print(err.filename)  # doctest: +ELLIPSIS
    ...     print(err.lineno)
    ...     print(err.offset)
    /...<doctest src.functional.ffront.func_to_foast.FieldOperatorParser[...]>
    2
    4
    """

    source: str
    filename: str
    starting_line: int
    closure_refs: ClosureRefs
    externals_defs: dict[str, Any]

    def _make_loc(self, node: ast.AST) -> SourceLocation:
        loc = SourceLocation.from_AST(node, source=self.filename)
        return SourceLocation(
            line=loc.line + self.starting_line - 1,
            column=loc.column,
            source=loc.source,
            end_line=loc.end_line + self.starting_line - 1,
            end_column=loc.end_column,
        )

    def _make_syntax_error(self, node: ast.AST, *, message: str = "") -> FieldOperatorSyntaxError:
        err = FieldOperatorSyntaxError.from_AST(
            node, msg=message, filename=self.filename, text=self.source
        )
        err.lineno = (err.lineno or 1) + self.starting_line - 1
        return err

    @classmethod
    def apply(
        cls,
        source_definition: SourceDefinition,
        closure_refs: ClosureRefs,
        externals_defs: Optional[dict[str, Any]] = None,
    ) -> foast.FieldOperator:
        source, filename, starting_line = source_definition
        result = None
        try:
            definition_ast = StringifyAnnotationsPass.apply(
                ast.parse(textwrap.dedent(source)).body[0]
            )
            ssa = SingleStaticAssignPass.apply(definition_ast)
            sat = SingleAssignTargetPass.apply(ssa)
            las = UnpackedAssignPass.apply(sat)
            result = cls(
                source=source,
                filename=filename,
                starting_line=starting_line,
                closure_refs=closure_refs,
                externals_defs=externals_defs,
            ).visit(las)
        except SyntaxError as err:
            if not err.filename:
                err.filename = filename
            if not isinstance(err, FieldOperatorSyntaxError):
                err.lineno = (err.lineno or 1) + starting_line - 1
            raise err

        return FieldOperatorShiftRecognition.apply(FieldOperatorTypeDeduction.apply(result))

    @classmethod
    def apply_to_function(
        cls,
        func: types.FunctionType,
        externals: Optional[dict[str, Any]] = None,
    ) -> foast.FieldOperator:
        source_definition = SourceDefinition.from_function(func)
        closure_refs = ClosureRefs.from_function(func)
        return cls.apply(source_definition, closure_refs, externals)

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs) -> foast.FieldOperator:
        _, _, imported_names, nonlocal_names, global_names = SymbolNames.from_source(
            self.source, self.filename
        )
        # TODO(egparedes): raise the exception at the first use of the undefined symbol
        if missing_defs := (self.closure_refs.unbound - imported_names):
            raise self._make_syntax_error(
                node, message=f"Missing symbol definitions: {missing_defs}"
            )

        # 'SymbolNames.from_source()' uses the symtable module to analyze the isolated source
        # code of the function, and thus all non-local symbols are classified as 'global'.
        # However, 'closure_refs' comes from inspecting the live function object, which might
        # have not been defined at a global scope, and therefore actual symbol values could appear
        # in both 'closure_refs.globals' and 'self.closure_refs.nonlocals'.
        defs = self.closure_refs.globals | self.closure_refs.nonlocals
        closure = [
            symbol_makers.make_symbol_from_value(
                name, defs[name], foast.Namespace.CLOSURE, self._make_loc(node)
            )
            for name in global_names | nonlocal_names
        ]

        return foast.FieldOperator(
            id=node.name,
            params=self.visit(node.args),
            body=self.visit_stmt_list(node.body),
            closure=closure,
            location=self._make_loc(node),
        )

    def visit_Import(self, node: ast.Import, **kwargs) -> None:
        raise self._make_syntax_error(
            node, f"Only 'from' imports from {fbuiltins.MODULE_BUILTIN_NAMES} are supported"
        )

    def visit_ImportFrom(self, node: ast.ImportFrom, **kwargs) -> None:
        if node.module not in fbuiltins.MODULE_BUILTIN_NAMES:
            raise self._make_syntax_error(
                node, f"Only 'from' imports from {fbuiltins.MODULE_BUILTIN_NAMES} are supported"
            )

        symbols = []

        if node.module == fbuiltins.EXTERNALS_MODULE_NAME:
            for alias in node.names:
                if alias.name not in self.externals_defs:
                    raise self._make_syntax_error(
                        node, message="Missing symbol '{alias.name}' definition in {node.module}}"
                    )
                symbols.append(
                    symbol_makers.make_symbol_from_value(
                        alias.asname or alias.name,
                        self.externals_defs[alias.name],
                        foast.Namespace.EXTERNAL,
                        location=self._make_loc(node),
                    )
                )

        return foast.ExternalImport(symbols=symbols, location=self._make_loc(node))

    def visit_arguments(self, node: ast.arguments) -> list[foast.FieldSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> foast.FieldSymbol:
        if (annotation := self.closure_refs.annotations.get(node.arg, None)) is None:
            raise self._make_syntax_error(node, message="Untyped parameters not allowed!")
        new_type = symbol_makers.make_symbol_type_from_typing(annotation)
        return foast.FieldSymbol(id=node.arg, location=self._make_loc(node), type=new_type)

    def visit_Assign(self, node: ast.Assign, **kwargs) -> foast.Assign:
        target = node.targets[0]  # there is only one element after assignment passes
        if isinstance(target, ast.Tuple):
            raise self._make_syntax_error(node, message="Unpacking not allowed!")
        if not isinstance(target, ast.Name):
            raise self._make_syntax_error(node, message="Can only assign to names!")
        new_value = self.visit(node.value)
        target_symbol_type = foast.FieldSymbol
        constraint_type = common_types.FieldType
        if isinstance(new_value, foast.TupleExpr):
            target_symbol_type = foast.TupleSymbol
            constraint_type = common_types.TupleType
        return foast.Assign(
            target=target_symbol_type(
                id=target.id,
                location=self._make_loc(target),
                type=common_types.DeferredSymbolType(constraint=constraint_type),
            ),
            value=new_value,
            location=self._make_loc(node),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign, **kwargs) -> foast.Assign:
        if not isinstance(node.target, ast.Name):
            raise self._make_syntax_error(node, message="Can only assign to names!")

        if node.annotation is not None:
            assert isinstance(
                node.annotation, ast.Constant
            ), "Annotations should be ast.Constant(string). Use StringifyAnnotationsPass"
            global_ns = {**fbuiltins.BUILTINS, **self.closure_refs.globals}
            local_ns = self.closure_refs.nonlocals
            annotation = eval(node.annotation.value, global_ns, local_ns)
            target_type = target_type = symbol_makers.make_symbol_type_from_typing(
                annotation, global_ns=global_ns, local_ns=local_ns
            )
        else:
            target_type = common_types.DeferredSymbolType()

        return foast.Assign(
            target=foast.FieldSymbol(
                id=node.target.id,
                location=self._make_loc(node.target),
                type=target_type,
            ),
            value=self.visit(node.value) if node.value else None,
            location=self._make_loc(node),
        )

    def visit_Subscript(self, node: ast.Subscript, **kwargs) -> foast.Subscript:
        if not isinstance(node.slice, ast.Constant):
            raise self._make_syntax_error(node, message="""Subscript slicing not allowed!""")
        return foast.Subscript(
            value=self.visit(node.value), index=node.slice.value, location=self._make_loc(node)
        )

    def visit_Tuple(self, node: ast.Tuple, **kwargs) -> foast.TupleExpr:
        return foast.TupleExpr(
            elts=[self.visit(item) for item in node.elts], location=self._make_loc(node)
        )

    def visit_Return(self, node: ast.Return, **kwargs) -> foast.Return:
        if not node.value:
            raise self._make_syntax_error(node, message="Empty return not allowed")
        return foast.Return(value=self.visit(node.value), location=self._make_loc(node))

    def visit_stmt_list(self, nodes: list[ast.stmt]) -> list[foast.Expr]:
        if not isinstance(last_node := nodes[-1], ast.Return):
            raise self._make_syntax_error(
                last_node, message="Field operator must return a field expression on the last line!"
            )
        return [self.visit(node) for node in nodes]

    def visit_Name(self, node: ast.Name, **kwargs) -> foast.Name:
        return foast.Name(id=node.id, location=self._make_loc(node))

    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs) -> foast.UnaryOp:
        return foast.UnaryOp(
            op=self.visit(node.op), operand=self.visit(node.operand), location=self._make_loc(node)
        )

    def visit_UAdd(self, node: ast.UAdd, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.UADD

    def visit_USub(self, node: ast.USub, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.USUB

    def visit_Not(self, node: ast.Not, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.NOT

    def visit_BinOp(self, node: ast.BinOp, **kwargs) -> foast.BinOp:
        new_op = None
        try:
            new_op = self.visit(node.op)
        except FieldOperatorSyntaxError as err:
            err.lineno = node.lineno
            err.offset = node.col_offset
            raise err
        return foast.BinOp(
            op=new_op,
            left=self.visit(node.left),
            right=self.visit(node.right),
            location=self._make_loc(node),
        )

    def visit_Add(self, node: ast.Add, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.MULT

    def visit_Div(self, node: ast.Div, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.DIV

    def visit_Pow(self, node: ast.Pow, **kwargs) -> None:
        raise FieldOperatorSyntaxError(
            msg="`**` operator not supported!",
        )

    def visit_Mod(self, node: ast.Mod, **kwargs) -> None:
        raise FieldOperatorSyntaxError(
            msg="`%` operator not supported!",
        )

    def visit_BitAnd(self, node: ast.BitAnd, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.BIT_AND

    def visit_BitOr(self, node: ast.BitOr, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.BIT_OR

    def visit_BoolOp(self, node: ast.BoolOp, **kwargs) -> None:
        try:
            self.visit(node.op)
        except FieldOperatorSyntaxError as err:
            err.lineno = node.lineno
            err.offset = node.col_offset
            raise err

    def visit_And(self, node: ast.And, **kwargs) -> None:
        raise FieldOperatorSyntaxError(msg="`and` operator not allowed!")

    def visit_Or(self, node: ast.Or, **kwargs) -> None:
        raise FieldOperatorSyntaxError(msg="`or` operator not allowed!")

    def visit_Compare(self, node: ast.Compare, **kwargs) -> foast.Compare:
        if len(node.comparators) == 1:
            return foast.Compare(
                op=self.visit(node.ops[0]),
                left=self.visit(node.left),
                right=self.visit(node.comparators[0]),
                location=self._make_loc(node),
            )
        smaller_node = copy.copy(node)
        smaller_node.comparators = node.comparators[1:]
        smaller_node.ops = node.ops[1:]
        smaller_node.left = node.comparators[0]
        return foast.Compare(
            op=self.visit(node.ops[0]),
            left=self.visit(node.left),
            right=self.visit(smaller_node),
            location=self._make_loc(node),
        )

    def visit_Gt(self, node: ast.Gt, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.GT

    def visit_Lt(self, node: ast.Lt, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.LT

    def visit_Eq(self, node: ast.Eq, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.EQ

    def visit_Call(self, node: ast.Call, **kwargs) -> foast.CompareOperator:
        new_func = self.visit(node.func)
        if not isinstance(new_func, foast.Name):
            raise self._make_syntax_error(
                node.func, message="functions can only be called directly!"
            )

        return foast.Call(
            func=new_func,
            args=[self.visit(arg) for arg in node.args],
            location=self._make_loc(node),
        )

    def generic_visit(self, node) -> None:
        raise self._make_syntax_error(node)


class FieldOperatorSyntaxError(common.GTSyntaxError):
    def __init__(
        self,
        msg="",
        *,
        lineno: int = 0,
        offset: int = 0,
        filename: Optional[str] = None,
        end_lineno: int = None,
        end_offset: int = None,
        text: Optional[str] = None,
    ):
        msg = f"Invalid Field Operator Syntax: {msg}"
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))

    @classmethod
    def from_AST(
        cls,
        node: ast.AST,
        *,
        msg: str = "",
        filename: Optional[str] = None,
        text: Optional[str] = None,
    ):
        return cls(
            msg,
            lineno=node.lineno,
            offset=node.col_offset,
            filename=filename,
            end_lineno=node.end_lineno,
            end_offset=node.end_col_offset,
        )
