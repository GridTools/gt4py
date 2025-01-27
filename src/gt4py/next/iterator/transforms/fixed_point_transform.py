# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
from typing import ClassVar, Optional, Type

from gt4py import eve
from gt4py.next.iterator import ir
from gt4py.next.iterator.type_system import inference as itir_type_inference


@dataclasses.dataclass(frozen=True, kw_only=True)
class FixedPointTransform(eve.PreserveLocationVisitor, eve.NodeTranslator):
    Flag: ClassVar[Type[enum.Flag]]
    flags: enum.Flag

    def fp_transform(self, node: ir.Node, **kwargs) -> ir.Node:
        while True:
            new_node = self.transform(node, **kwargs)
            if new_node is None:
                break
            assert new_node != node
            node = new_node
        return node

    def transform(self, node: ir.Node, **kwargs) -> Optional[ir.Node]:
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
