# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import types

import numpy as np

import gt4py.next as gtx
import gt4py.next.ffront.field_operator_ast as foast
from gt4py.next import Dims, Dimension, float64, where
from gt4py.next.ffront.foast_passes import closure_var_folding
from gt4py.next.ffront.func_to_foast import FieldOperatorParser


I = Dimension("I")
IField = gtx.Field[Dims[I], float64]
BField = gtx.Field[Dims[I], bool]


@gtx.field_operator
def helper(a: IField) -> IField:
    return a + 1.0


helper_alias = helper
my_where = where
helpers = types.ModuleType("helpers")
helpers.helper = helper
more_helpers = types.ModuleType("more_helpers")
more_helpers.also_helper = helper


def _parse(func):
    return FieldOperatorParser.apply_to_function(func)


def _callee(node) -> str:
    return node.pre_walk_values().if_isinstance(foast.Call).to_list()[0].func.id


def _closure_symbol_ids(node) -> set[str]:
    return {symbol.id for symbol in node.closure_vars}


def test_direct_reference_is_kept():
    def testee(a: IField) -> IField:
        return helper(a)

    assert _callee(_parse(testee)) == "helper"


def test_aliased_reference_is_kept():
    def testee(a: IField) -> IField:
        return helper_alias(a)

    assert _callee(_parse(testee)) == "helper_alias"


def test_module_prefixed_builtin_is_canonicalized():
    def testee(cond: BField, a: IField, b: IField) -> IField:
        return gtx.where(cond, a, b)

    node = _parse(testee)
    assert _callee(node) == "where"
    assert "where" in _closure_symbol_ids(node)


def test_aliased_builtin_is_canonicalized():
    def testee(cond: BField, a: IField, b: IField) -> IField:
        return my_where(cond, a, b)

    node = _parse(testee)
    assert _callee(node) == "where"
    assert "where" in _closure_symbol_ids(node)


def test_module_prefixed_operator_gets_a_synthesized_name():
    def testee(a: IField) -> IField:
        return helpers.helper(a)

    def other_testee(a: IField) -> IField:
        return more_helpers.also_helper(a)

    name = closure_var_folding._operator_name(helper)
    assert name.startswith("helper_") and name.isidentifier()
    for func in (testee, other_testee):
        node = _parse(func)
        assert _callee(node) == name
        assert name in _closure_symbol_ids(node)


def test_module_scalar_attribute_is_folded():
    def testee(a: IField) -> IField:
        return a * np.pi

    constants = _parse(testee).pre_walk_values().if_isinstance(foast.Constant).to_list()
    assert [c.value for c in constants] == [np.pi]
