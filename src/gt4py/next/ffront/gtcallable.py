# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import abc
import typing
from typing import Any, Optional

from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts


@typing.runtime_checkable
class GTCallable(typing.Protocol):
    """
    Typing Protocol (abstract base class) defining the interface for subroutines.

    Any class implementing the methods defined in this protocol can be called
    from ``ffront`` programs or operators.
    """

    def __gt_closure_vars__(self) -> Optional[dict[str, Any]]:
        """
        Return all external variables referenced inside the callable.

        Note that in addition to the callable itself all captured variables
        are also lowered such that they can be used in the lowered callable.
        """
        return None

    @abc.abstractmethod
    def __gt_type__(self) -> ts.CallableType:
        """
        Return symbol type, i.e. signature and return type.

        The type is used internally to populate the closure vars of the
        various dialects root nodes (i.e. FOAST Field Operator, PAST Program)
        """
        ...

    @abc.abstractmethod
    def __gt_itir__(self) -> itir.FunctionDefinition:
        """
        Return iterator IR function definition representing the callable.

        Used internally by the Program decorator to populate the function
        definitions of the iterator IR.
        """
        ...

    # FIXME[#1582](tehrengruber): remove after refactoring to GTIR
    @abc.abstractmethod
    def __gt_gtir__(self) -> itir.FunctionDefinition:
        """
        Return iterator IR function definition representing the callable.

        Used internally by the Program decorator to populate the function
        definitions of the iterator IR.
        """
        ...

    # TODO(tehrengruber): For embedded execution a `__call__` method and for
    #  "truly" embedded execution arguably also a `from_function` method is
    #  required. Since field operators currently have a `__gt_type__` with a
    #  Field return value, but it's `__call__` method being void (result via
    #  out arg) there is no good / consistent definition on what signature a
    #  protocol implementer is expected to provide. Skipping for now.
