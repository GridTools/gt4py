# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

import typing

from eve import extended_typing as xtyping
from eve.extended_typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    ForwardRef,
    FrozenSet,
    List,
    Set,
    Tuple,
    Type,
)


def test_is_protocol():
    class AProtocol(typing.Protocol):
        def do_something(self, value: int) -> int:
            ...

    class NotProtocol(AProtocol):
        def do_something_else(self, value: float) -> float:
            ...

    class AXProtocol(xtyping.Protocol):
        A = 1

    class NotXProtocol(AXProtocol):
        A = 1

    class AgainProtocol(AProtocol, xtyping.Protocol):
        def do_something_else(self, value: float) -> float:
            ...

    assert xtyping.is_protocol(AProtocol)
    assert xtyping.is_protocol(AXProtocol)

    assert not xtyping.is_protocol(NotProtocol)
    assert not xtyping.is_protocol(NotXProtocol)

    assert xtyping.is_protocol(AgainProtocol)


def test_get_partial_type_hints():
    def f1(a: int) -> float:
        ...

    assert xtyping.get_partial_type_hints(f1) == {"a": int, "return": float}

    class MissingRef:
        ...

    def f_partial(a: int) -> MissingRef:
        ...

    # This is expected behavior because this test file uses
    # 'from __future__ import annotations' and therefore local
    # references cannot be automatically resolved
    assert xtyping.get_partial_type_hints(f_partial) == {
        "a": int,
        "return": ForwardRef("MissingRef"),
    }
    assert xtyping.get_partial_type_hints(f_partial, localns={"MissingRef": MissingRef}) == {
        "a": int,
        "return": MissingRef,
    }
    assert xtyping.get_partial_type_hints(f_partial, globalns={"MissingRef": int}) == {
        "a": int,
        "return": int,
    }

    def f_nested_partial(a: int) -> Dict[str, MissingRef]:
        ...

    assert xtyping.get_partial_type_hints(f_nested_partial) == {
        "a": int,
        "return": ForwardRef("Dict[str, MissingRef]"),
    }
    assert xtyping.get_partial_type_hints(f_nested_partial, localns={"MissingRef": MissingRef}) == {
        "a": int,
        "return": Dict[str, MissingRef],
    }

    def f_annotated(a: Annotated[int, "Foo"]) -> float:
        ...

    assert xtyping.get_partial_type_hints(f_annotated) == {"a": int, "return": float}
    assert xtyping.get_partial_type_hints(f_annotated, include_extras=True) == {
        "a": Annotated[int, "Foo"],
        "return": float,
    }
    assert xtyping.get_partial_type_hints(f_annotated, include_extras=True) != {
        "a": Annotated[int, "Bar"],
        "return": float,
    }


def test_eval_forward_ref():
    assert xtyping.eval_forward_ref("Dict[str, Tuple[int, float]]") == Dict[str, Tuple[int, float]]
    assert (
        xtyping.eval_forward_ref(ForwardRef("Dict[str, Tuple[int, float]]"))
        == Dict[str, Tuple[int, float]]
    )

    class MissingRef:
        ...

    assert (
        xtyping.eval_forward_ref("Callable[[int], MissingRef]", localns={"MissingRef": MissingRef})
        == Callable[[int], MissingRef]
    )

    assert (
        xtyping.eval_forward_ref(
            "Callable[[int], MissingRef]",
            globalns={"Callable": Callable},
            localns={"MissingRef": MissingRef},
        )
        == Callable[[int], MissingRef]
    )

    assert (
        xtyping.eval_forward_ref(
            "Callable[[Annotated[int, 'Foo']], MissingRef]",
            globalns={"Annotated": Annotated, "Callable": Callable},
            localns={"MissingRef": MissingRef},
        )
        == Callable[[int], MissingRef]
    )
    assert (
        xtyping.eval_forward_ref(
            "Callable[[Annotated[int, 'Foo']], MissingRef]",
            globalns={"Annotated": Annotated, "Callable": Callable},
            localns={"MissingRef": MissingRef},
            include_extras=True,
        )
        == Callable[[Annotated[int, "Foo"]], MissingRef]
    )


def test_infer_type():
    assert xtyping.infer_type(3) == int

    assert xtyping.infer_type(None) is type(None)  # noqa: E721  # do not compare types
    assert xtyping.infer_type(type(None)) is type(None)  # noqa: E721  # do not compare types
    assert xtyping.infer_type(None, none_as_type=False) is None
    assert xtyping.infer_type(type(None), none_as_type=False) is None

    assert xtyping.infer_type(Dict[str, int]) == Dict[str, int]

    assert xtyping.infer_type({1, 2, 3}) == Set[int]
    assert xtyping.infer_type(frozenset({"1", "2", "3"})) == FrozenSet[str]

    assert xtyping.infer_type({"a": [0], "b": [1]}) == Dict[str, List[int]]

    assert xtyping.infer_type(str) == Type[str]

    class A:
        ...

    assert xtyping.infer_type(A()) == A
    assert xtyping.infer_type(A) == Type[A]

    def f1():
        ...

    assert xtyping.infer_type(f1) == Callable[[], Any]

    def f2(a: int, b: float) -> None:
        ...

    assert xtyping.infer_type(f2) == Callable[[int, float], type(None)]

    def f3(
        a: Dict[Tuple[str, ...], List[int]],
        b: List[Callable[[List[int]], Set[Set[int]]]],
        c: Type[List[int]],
    ) -> Any:
        ...

    assert (
        xtyping.infer_type(f3)
        == Callable[
            [
                Dict[Tuple[str, ...], List[int]],
                List[Callable[[List[int]], Set[Set[int]]]],
                Type[List[int]],
            ],
            Any,
        ]
    )

    def f4(a: int, b: float, *, foo: Tuple[str, ...] = ()) -> None:
        ...

    assert xtyping.infer_type(f4) == Callable[[int, float], type(None)]
    assert (
        xtyping.infer_type(f4, annotate_callable_kwargs=True)
        == Annotated[
            Callable[[int, float], type(None)], xtyping.CallableKwargsInfo({"foo": Tuple[str, ...]})
        ]
    )
