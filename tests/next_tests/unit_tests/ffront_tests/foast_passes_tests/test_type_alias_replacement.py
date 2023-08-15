# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from typing import TypeAlias
import typing

import ast
import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import astype
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.ffront.dialect_parser import parse_source_definition
from gt4py.next import float32, float64
from gt4py.next.ffront.foast_passes.type_alias_replacement import TypeAliasReplacement
from gt4py.next.ffront.ast_passes.fix_missing_locations import FixMissingLocations
from gt4py.next.ffront.ast_passes.remove_docstrings import RemoveDocstrings
from gt4py.next.ffront.source_utils import SourceDefinition, get_closure_vars_from_function
from gt4py.next.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    StringifyAnnotationsPass,
    UnchainComparesPass,
)

TDim = gtx.Dimension("TDim")  # Meaningless dimension, used for tests.

def test_type_alias_replacement_vpfloat():
    vpfloat: TypeAlias = float32

 #  Parse a function 
    def fieldOp_with_TypeAlias(a: gtx.Field[[TDim], vpfloat]) -> gtx.Field[[TDim], vpfloat] :
        return vpfloat("3.1418") + astype(a, vpfloat)
    
    # Steps of apply_to_function() from DialectParser
    src = SourceDefinition.from_function(fieldOp_with_TypeAlias)
    closure_vars = get_closure_vars_from_function(fieldOp_with_TypeAlias)
    annotations = typing.get_type_hints(fieldOp_with_TypeAlias)

    # Steps of apply() from DialectParser
    definition_ast: ast.AST
    definition_ast = parse_source_definition(src)
    definition_ast = RemoveDocstrings.apply(definition_ast)
    definition_ast = FixMissingLocations.apply(definition_ast)

    # Steps of _preprocess_definition_ast() from FieldOperatorParser
    sta = StringifyAnnotationsPass.apply(definition_ast)
    ssa = SingleStaticAssignPass.apply(sta)
    sat = SingleAssignTargetPass.apply(ssa)
    ucc = UnchainComparesPass.apply(sat)

    # Instance creation
    foast_node = FieldOperatorParser(source_definition=src,
                                     closure_vars=closure_vars,
                                     annotations=annotations)
    
    # Apply TypeAliasReplacement (func to test)
    foast_node = TypeAliasReplacement.apply(foast_node.visit(ucc), closure_vars)[0]
   # foast_node = FieldOperatorTypeDeduction.apply(foast_node)

    assert (
            foast_node.body.stmts[0].value.left.func.id == "float32" and
            foast_node.body.stmts[0].value.right.args[1].id == "float32"
    )

def test_type_alias_replacement_wpfloat():
    wpfloat: TypeAlias = float64

    #  Parse a function
    def fieldOp_with_TypeAlias(a: gtx.Field[[TDim], wpfloat]) -> gtx.Field[[TDim], wpfloat]:
        return wpfloat("3.1418") + astype(a, wpfloat)

    # Steps of apply_to_function() from DialectParser
    src = SourceDefinition.from_function(fieldOp_with_TypeAlias)
    closure_vars = get_closure_vars_from_function(fieldOp_with_TypeAlias)
    annotations = typing.get_type_hints(fieldOp_with_TypeAlias)

    # Steps of apply() from DialectParser
    definition_ast: ast.AST
    definition_ast = parse_source_definition(src)
    definition_ast = RemoveDocstrings.apply(definition_ast)
    definition_ast = FixMissingLocations.apply(definition_ast)

    # Steps of _preprocess_definition_ast() from FieldOperatorParser
    sta = StringifyAnnotationsPass.apply(definition_ast)
    ssa = SingleStaticAssignPass.apply(sta)
    sat = SingleAssignTargetPass.apply(ssa)
    ucc = UnchainComparesPass.apply(sat)

    # Instance creation
    foast_node = FieldOperatorParser(source_definition=src,
                                     closure_vars=closure_vars,
                                     annotations=annotations)

    # Apply TypeAliasReplacement (func to test)
    foast_node = TypeAliasReplacement.apply(foast_node.visit(ucc), closure_vars)[0]
    # foast_node = FieldOperatorTypeDeduction.apply(foast_node)

    assert (
            foast_node.body.stmts[0].value.left.func.id == "float64" and
            foast_node.body.stmts[0].value.right.args[1].id == "float64"
    )
