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


class FixedPointTransformation(eve.NodeTranslator):
    """
    Base class for iterative transformations that converge when a fixed-point is reached.
    """

    REINFER_TYPES: ClassVar[bool] = False

    def visit(self, node, **kwargs):
        node = super().visit(node, **kwargs)
        return self.fp_transform(node, **kwargs) if isinstance(node, ir.Node) else node

    def fp_transform(self, node: ir.Node, **kwargs) -> ir.Node:
        """
        Transform node until a fixed point is reached, e.g. no transformation is applicable anymore.
        """
        while True:
            new_node = self.transform(node, **kwargs)
            if new_node is None:
                break
            else:
                new_node = self.post_transform(node, new_node)
            assert new_node != node
            node = new_node
        return node

    def post_transform(self, node: ir.Node, new_node: ir.Node) -> ir.Node:
        if self.REINFER_TYPES:
            itir_type_inference.reinfer(new_node)
        self._preserve_annex(node, new_node)
        return new_node

    """
    Transform node once.
    
    Execute transformation if applicable. When a transformation occurred the function will return 
    the transformed node. Note that the transformation itself may call other transformations on 
    child nodes again.
    """

    def transform(self, node: ir.Node, **kwargs) -> Optional[ir.Node]: ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class CombinedFixedPointTransform(FixedPointTransformation):
    """
    Base class for a set of iterative transformations that converge when a fixed-point is reached.
    """

    #: Enum of all transformation (names). The transformations need to be defined as methods
    #: named `transform_<NAME>`.
    Transformation: ClassVar[Type[enum.Flag]]

    #: All transformations enabled in this instance, e.g. `Transformation.T1 & Transformation.T2`.
    #: Usually the default value is chosen to be all transformations.
    enabled_transformations: enum.Flag

    def transform(self, node: ir.Node, **kwargs) -> Optional[ir.Node]:
        for transformation in self.Transformation:
            if self.enabled_transformations & transformation:
                assert isinstance(transformation.name, str)
                method = getattr(self, f"transform_{transformation.name.lower()}")
                result = method(node, **kwargs)
                if result is not None:
                    assert result is not node, (
                        f"Transformation {transformation.name.lower()} should have returned None, since nothing changed."
                    )
                    if self.REINFER_TYPES:
                        itir_type_inference.reinfer(result)
                    self._preserve_annex(node, result)
                    return result
        return None
