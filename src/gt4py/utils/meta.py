# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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

"""Utilities for runtime: introspection and bytecode."""

import ast
import copy
import inspect
import operator
import platform
import textwrap

from packaging import version

from .base import shashed_id


def get_closure(func, *, include_globals=True, included_nonlocals=True, include_builtins=True):
    closure_vars = inspect.getclosurevars(func)
    closure = {}
    unbound = set(closure_vars.unbound)
    if include_globals:
        closure.update(closure_vars.globals)
    else:
        unbound |= set(closure_vars.globals.keys())
    if included_nonlocals:
        closure.update(closure_vars.nonlocals)
    else:
        unbound |= set(closure_vars.nonlocals.keys())
    if include_builtins:
        closure.update(closure_vars.builtins)
    else:
        unbound |= set(closure_vars.builtins.keys())

    return closure, unbound


def get_source(func):
    try:
        source = inspect.getsource(func)
    except OSError as e:
        # Dynamically generated functions using exec() do not contain references
        # to the original source code, so we can use the convention of attaching
        # the source to the function using the "__exec_source__" attribute
        source = getattr(func, "__exec_source__", None)
        if not isinstance(source, str):
            raise e

    return source


def get_ast(func_or_source_or_ast):
    if callable(func_or_source_or_ast):
        func_or_source_or_ast = get_source(func_or_source_or_ast)
    if isinstance(func_or_source_or_ast, str):
        func_or_source_or_ast = ast.parse(
            textwrap.dedent(func_or_source_or_ast), feature_version=(3, 9)
        )
    if isinstance(func_or_source_or_ast, (ast.AST, list)):
        ast_root = func_or_source_or_ast
    else:
        raise ValueError("Invalid function definition ({})".format(func_or_source_or_ast))
    return ast_root


def ast_dump(definition, *, skip_annotations: bool = True, skip_decorators: bool = True) -> str:
    def _dump(node: ast.AST, excluded_names):
        if isinstance(node, ast.AST):
            if excluded_names:
                fields = [
                    (name, _dump(value, excluded_names))
                    for name, value in sorted(ast.iter_fields(node))
                    if name not in excluded_names
                ]
            else:
                fields = [
                    (name, _dump(value, excluded_names))
                    for name, value in sorted(ast.iter_fields(node))
                ]

            return "".join(
                [
                    node.__class__.__name__,
                    "({content})".format(
                        content=", ".join("{}={}".format(name, value) for name, value in fields)
                    ),
                ]
            )

        elif isinstance(node, list):
            lines = ["[", *[_dump(i, excluded_names) + "," for i in node], "]"]
            return "\n".join(lines)

        else:
            return repr(node)

    skip_node_names = set()
    if skip_decorators:
        skip_node_names.add("decorator_list")
    if skip_annotations:
        skip_node_names.add("annotation")

    dumped_ast = _dump(get_ast(definition), skip_node_names)

    return dumped_ast


def ast_unparse(ast_node):
    """Call ast.unparse, but use astunparse for Python prior to 3.9."""
    if version.parse(platform.python_version()) < version.parse("3.9"):
        import astunparse

        return astunparse.unparse(ast_node)
    else:
        return ast.unparse(ast_node)


def ast_shash(ast_node, *, skip_decorators=True):
    return shashed_id(ast_dump(ast_node, skip_decorators=skip_decorators))


def collect_decorators(func_or_source_or_ast):
    if callable(func_or_source_or_ast):
        func_or_source_or_ast = get_source(func_or_source_or_ast)
    assert isinstance(func_or_source_or_ast, (str, ast.FunctionDef))

    if isinstance(func_or_source_or_ast, str):
        source = textwrap.dedent(func_or_source_or_ast)
        lines = source.splitlines()
        result = []
        i = 0
        while not lines[i].startswith("def "):
            if lines[i].startswith("@"):
                end = lines[i].find("(")
                if end < 0:
                    end = len(lines[i]) - 1
                decorator = lines[i][1:end]
                result.append(decorator)
            i += 1
    else:
        result = func_or_source_or_ast.decorator_list

    return result


def remove_decorators(func_or_source_or_ast):
    if callable(func_or_source_or_ast):
        func_or_source_or_ast = get_source(func_or_source_or_ast)
    assert isinstance(func_or_source_or_ast, (str, ast.FunctionDef))

    if isinstance(func_or_source_or_ast, str):
        source = textwrap.dedent(func_or_source_or_ast)
        lines = source.splitlines()
        start = 0
        for n, line in enumerate(lines):
            if line.startswith("def "):
                start = n
                break
        result = "\n".join(lines[start:])
    else:
        ast_root = copy.copy(func_or_source_or_ast)
        ast_root.decorator_list = []
        result = ast_root

    return result


def split_def_decorators(func_or_source):
    if callable(func_or_source):
        func_or_source = get_source(func_or_source)
    assert isinstance(func_or_source, str)

    source = textwrap.dedent(func_or_source)
    lines = source.splitlines()
    start = 0
    for n, line in enumerate(lines):
        if line.startswith("def "):
            start = n
            break
    definition = "\n".join(lines[start:])
    decorators = "\n".join(lines[:start])

    return definition, decorators


def get_qualified_name_from_node(name_or_attribute, *, as_list=False):
    assert isinstance(name_or_attribute, (ast.Attribute, ast.Name))

    node = name_or_attribute
    if isinstance(node, ast.Name):
        components = [node.id]
    else:
        assert isinstance(node, ast.Attribute)
        components = get_qualified_name_from_node(node.value, as_list=True)
        components = components + [node.attr]

    return components if as_list else ".".join(components)


class ASTPass:
    """Clone of the ast.NodeVisitor that supports forwarding kwargs."""

    def __call__(self, func_or_source_or_ast):
        ast_root = get_ast(func_or_source_or_ast)
        return self.visit(ast_root)

    def visit(self, node, **kwargs):
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, **kwargs)

    def generic_visit(self, node, **kwargs):
        """Fallback if no explicit visitor function exists for a node."""
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item, **kwargs)
            elif isinstance(value, ast.AST):
                self.visit(value, **kwargs)


class ASTTransformPass(ASTPass):
    """Clone of the ast.NodeTransformer that supports forwarding kwargs."""

    def generic_visit(self, node, **kwargs):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value, **kwargs)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value, **kwargs)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class ASTEvaluator(ASTPass):
    AST_OP_TO_OP = {
        # Arithmetic operations
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        # Bitwise operations
        ast.Invert: operator.invert,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        # Logical operations
        ast.Not: operator.not_,
        # ast.And: lambda a, b: a and b,
        # ast.Or: lambda a, b: a or b,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    @classmethod
    def apply(cls, func_or_source_or_ast, context, default=None):
        try:
            result = cls(context)(func_or_source_or_ast)

        except (KeyError, ValueError):
            result = default
        return result

    def __init__(self, context: dict):
        self.context = context

    def __call__(self, func_or_source_or_ast):
        return super().__call__(func_or_source_or_ast)

    def visit_Name(self, node):
        return self.context[node.id]

    def visit_Num(self, node):
        return node.n

    def visit_Constant(self, node):
        return node.n

    def visit_NameConstant(self, node):
        return node.value

    def visit_Tuple(self, node: ast.Tuple):
        return tuple(self.visit(elem) for elem in node.elts)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        val = self.visit(node.operand)
        return self.AST_OP_TO_OP[type(node.op)](val)

    def visit_BinOp(self, node: ast.BinOp):
        return self.AST_OP_TO_OP[type(node.op)](self.visit(node.left), self.visit(node.right))

    def visit_BoolOp(self, node: ast.BoolOp):
        # Use short-circuited evaluation of logical expressions
        condition = True if isinstance(node.op, ast.And) else False
        for value in node.values:
            if bool(self.visit(value)) is not condition:
                return not condition

        return condition

    def visit_Attribute(self, node: ast.Attribute):
        qualified_name = get_qualified_name_from_node(node)
        if qualified_name not in self.context:
            raise ValueError(f"{qualified_name} not found in context")
        return self.context[qualified_name]

    def visit_Compare(self, node: ast.Compare):
        values = [self.visit(node.left)] + [self.visit(cmp) for cmp in node.comparators]
        comparisons = [
            self.AST_OP_TO_OP[type(op)](values[i], values[i + 1]) for i, op in enumerate(node.ops)
        ]
        return all(comparisons)

    def generic_visit(self, node):
        raise ValueError("Invalid AST node for evaluation: {}".format(repr(node)))


ast_eval = ASTEvaluator.apply


class FunctionDefCollector(ASTPass):
    @classmethod
    def apply(cls, func_or_source_or_ast, max_defs=None):
        collector = cls(max_defs=max_defs)
        return collector(func_or_source_or_ast)

    def __init__(self, *, max_defs=None):
        self.max_defs = max_defs
        self.defs = None

    def __call__(self, func_or_source_or_ast):
        self.defs = []
        super().__call__(func_or_source_or_ast)
        return self.defs

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.max_defs is None or len(self.defs) < self.max_defs:
            self.defs.append(node)


collect_function_defs = FunctionDefCollector.apply


class AssignTargetsCollector(ASTPass):
    @classmethod
    def apply(cls, func_or_source_or_ast, *, allow_multiple_targets=True):
        collector = cls(allow_multiple_targets=allow_multiple_targets)
        return collector(func_or_source_or_ast)

    def __init__(self, *, allow_multiple_targets):
        self.assign_targets = []
        self.allow_multiple_targets = allow_multiple_targets

    def __call__(self, func_or_source_or_ast):
        self.defs = []
        super().__call__(func_or_source_or_ast)
        return self.assign_targets

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) > 1 and not self.allow_multiple_targets:
            raise RuntimeError(f"Multiple targets found in assignment ({node})")

        for target in node.targets:
            if isinstance(target, ast.Tuple):
                for t in target.elts:
                    assert isinstance(t, ast.Name)
                    self.assign_targets.append(t)
            else:
                self.assign_targets.append(target)


collect_assign_targets = AssignTargetsCollector.apply


class QualifiedNameCollector(ASTPass):
    @classmethod
    def apply(
        cls, func_or_source_or_ast, prefixes=None, *, skip_decorators=True, skip_annotations=True
    ):
        collector = cls(
            prefixes=prefixes, skip_decorators=skip_decorators, skip_annotations=skip_annotations
        )
        return collector(func_or_source_or_ast)

    def __init__(self, prefixes=None, *, skip_decorators=True, skip_annotations=True):
        self.prefixes = set(prefixes) if prefixes else None
        self.skip_decorators = skip_decorators
        self.skip_annotations = skip_annotations
        self.name_nodes = None

    def __call__(self, func_or_source_or_ast):
        self.name_nodes = {}
        super().__call__(func_or_source_or_ast)
        return self.name_nodes

    def generic_visit(self, node):
        if self.skip_decorators:
            node._fields = [name for name in node._fields if name != "decorator_list"]
        if self.skip_annotations:
            node._fields = [name for name in node._fields if name != "annotation"]
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        components, valid = self._get_name_components(node)
        if valid and not any(item is None for item in components):
            qualified_name = ".".join(components)
            self.name_nodes.setdefault(qualified_name, [])
            self.name_nodes[qualified_name].append(node)

    def visit_Name(self, node: ast.Name):
        if self.prefixes is None or node.id in self.prefixes:
            self.name_nodes.setdefault(node.id, [])
            self.name_nodes[node.id].append(node)

    def _get_name_components(self, node: ast.AST):
        if isinstance(node, ast.Name):
            components = [node.id]
            valid = self.prefixes is None or node.id in self.prefixes
        elif isinstance(node, ast.Attribute):
            components, valid = self._get_name_components(node.value)
            components = components + [node.attr]
            valid = valid or ".".join(components) in self.prefixes
        else:
            components = [None]
            valid = False

        return components, valid


collect_names = QualifiedNameCollector.apply


class ReturnCollector(ASTPass):
    @classmethod
    def apply(cls, func_or_source_or_ast):
        collector = cls()
        return collector(func_or_source_or_ast)

    def __init__(self):
        self.returns = None

    def __call__(self, func_or_source_or_ast):
        self.returns = []
        super().__call__(func_or_source_or_ast)
        return self.returns

    def visit_Return(self, node: ast.Return):
        self.returns.append(node)


collect_return_stmts = ReturnCollector.apply


class ImportsCollector(ASTPass):
    @classmethod
    def apply(cls, func_or_source_or_ast):
        collector = cls()
        return collector(func_or_source_or_ast)

    def __init__(self):
        self.bare_imports = None
        self.from_imports = None
        self.relative_imports = None

    def __call__(self, func_or_source_or_ast):
        self.bare_imports = {}
        self.from_imports = {}
        self.relative_imports = {}
        super().__call__(func_or_source_or_ast)
        return self.bare_imports, self.from_imports, self.relative_imports

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name
            as_name = alias.asname if alias.asname else name
            self.bare_imports[name] = as_name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module is None:
            imports_dict = self.relative_imports
            module = "".join(["."] * node.level)
        else:
            imports_dict = self.from_imports
            module = node.module

        for alias in node.names:
            name = ".".join([module, alias.name])
            as_name = alias.asname if alias.asname else name
            imports_dict[name] = as_name


collect_imported_symbols = ImportsCollector.apply


class SymbolsNameMapper(ASTTransformPass):
    @classmethod
    def apply(cls, func_or_source_or_ast, mapping, template_fmt=None, skip_names=None):
        collector = cls(mapping=mapping, template_fmt=template_fmt, skip_names=skip_names)
        return collector(func_or_source_or_ast)

    def __init__(self, mapping, template_fmt, skip_names):
        self.mapping = mapping
        self.template_fmt = template_fmt
        self.skip_names = set(skip_names)

    def __call__(self, func_or_source_or_ast):
        self.returns = []
        super().__call__(func_or_source_or_ast)
        return self.returns

    def visit_Name(self, node: ast.Name):
        self.generic_visit(node)
        if not self.skip_names or node.id not in self.skip_names:
            if node.id in self.mapping:
                node.id = self.mapping[node.id]
            else:
                node.id = self.template_fmt.format(name=node.id)
        return node


map_symbol_names = SymbolsNameMapper.apply
