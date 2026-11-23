# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import textwrap
import typing

import pytest

from gt4py.next import common
from gt4py.next.ffront import experimental, fbuiltins
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")


@pytest.mark.parametrize("tuple_spelling", [typing.Tuple, tuple, typing.Tuple[int], tuple[int]])
def test_type_conversion_helper_accepts_both_tuple_spellings(tuple_spelling):
    assert fbuiltins._type_conversion_helper(tuple_spelling) is ts.TupleType


@pytest.mark.parametrize(
    "union",
    [
        typing.Union[common.Field, typing.Tuple],
        # PEP 604 builds a 'types.UnionType', which has no '__origin__' at all
        common.Field | tuple,
    ],
)
def test_type_conversion_helper_accepts_both_union_spellings(union):
    assert fbuiltins._type_conversion_helper(union) == (ts.FieldType, ts.TupleType)


_DECLARED_BUILTINS = [(fbuiltins, name) for name in fbuiltins.FUN_BUILTIN_NAMES] + [
    (experimental, name) for name in experimental.EXPERIMENTAL_FUN_BUILTIN_NAMES
]


@pytest.mark.parametrize(
    "module, name", _DECLARED_BUILTINS, ids=[name for _, name in _DECLARED_BUILTINS]
)
def test_builtin_declaration(module, name):
    builtin = getattr(module, name)

    assert builtin.name == f"{module.__name__}.{name}"
    assert isinstance(builtin.__gt_type__(), ts.FunctionType)


def test_builtin_docstring():
    assert experimental.concat_where.__doc__.lstrip().startswith("Assemble")


def test_call_goes_to_the_implementation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fbuiltins, "call_implementation", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    fbuiltins.where("mask", "a", "b")
    fbuiltins.neighbor_sum("field", axis=IDim)

    assert calls == [
        ((fbuiltins.where, "mask", "a", "b"), {}),
        ((fbuiltins.neighbor_sum, "field"), {"axis": IDim}),
    ]


def test_call_without_implementation(monkeypatch):
    monkeypatch.setattr(fbuiltins, "call_implementation", fbuiltins._not_registered)

    with pytest.raises(RuntimeError, match=r"'gt4py\.next\.ffront\.fbuiltins\.sqrt' was called"):
        fbuiltins.sqrt(1.0)


def test_fbuiltins_does_not_import_embedded():
    """Importing `fbuiltins` loads no embedded module (function-local imports are not seen)."""
    # `gt4py.next` imports embedded itself, so it is replaced by a bare package of the same path
    code = textwrap.dedent(
        """
        import pathlib, sys, types
        import gt4py

        package = types.ModuleType("gt4py.next")
        package.__path__ = [str(pathlib.Path(gt4py.__file__).parent / "next")]
        sys.modules["gt4py.next"] = package
        import gt4py.next.ffront.fbuiltins

        print(sorted(name for name in sys.modules if name.startswith("gt4py.next.embedded")))
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
