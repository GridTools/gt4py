#!/usr/bin/env python
# -*- coding: utf-8 -*-

from eve import Node
from eve.traits import SymbolName, SymbolTableTrait
from eve.type_definitions import SymbolRef


class Sym(Node):
    id: SymbolName  # noqa: A003


class Expr(Node):
    ...


class Stmt(Node):
    ...


class SymRef(Expr):
    id: SymbolRef  # noqa: A003


class Return(Stmt):
    value: Expr


class Name(Expr):
    id: SymbolName  # noqa: A003


class FieldOperator(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[Sym]
    body: list[Stmt]
