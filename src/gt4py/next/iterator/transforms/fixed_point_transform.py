# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import functools
import operator
from typing import Optional, Type

from abc import abstractmethod

from gt4py import eve
from gt4py.next.iterator import ir
from gt4py.next.iterator.type_system import inference as itir_type_inference


@dataclasses.dataclass(frozen=True)
class FixedPointTransform(eve.PreserveLocationVisitor, eve.NodeTranslator):
    @property
    @abstractmethod
    def Flag(self) -> Type[enum.Flag]:
        pass

    @property
    @abstractmethod
    def flags(self) -> enum.Flag:
        pass

    def fp_transform(self, node: ir.Node, **kwargs) -> ir.Node:
        while True:
            new_node = self.transform(node, **kwargs)
            if new_node is None:
                break
            assert new_node != node
            node = new_node
        return node

    def transform(self, node: ir.Node, **kwargs) -> Optional[ir.Node]:
        if not isinstance(node, ir.FunCall):
            return None

        for transformation in self.Flag:
            if self.flags & transformation:
                assert isinstance(transformation.name, str)
                method = getattr(self, f"transform_{transformation.name.lower()}")
                result = method(node, **kwargs)
                if result is not None:
                    assert (
                            result is not node
                    )  # transformation should have returned None, since nothing changed
                    itir_type_inference.reinfer(result)
                    return result
        return None
