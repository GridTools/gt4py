# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

import types
from collections.abc import Mapping
from typing import Any, Union

from functional import common
from functional.ffront import func_to_foast as ff


def field_operator(
    definition: Union[
        types.FunctionType, ff.SourceDefinition, str, tuple[str, str], tuple[str, str, int], None
    ] = None,
    *,
    backend: common.Backend,
    externals: Mapping[str, Any] | None = None,
) -> common.FieldOperator:
    """
    Create a new GT4Py FieldOperator from the definition.

    Args:
        definition: Field operator definition. It can be either a Python function object or
            a source code string (optionally provided together with a virtual filename and
            a starting line number, which will used in syntax error messages).

    Keyword Args:
        backend: ``Backend`` object used for the implementation.
        externals: Extra symbol definitions used in the generation.

    Note:
        The values of ``externals`` symbols will only be evaluated during the generation
        of the new field operator. Subsequent changes in the associated values will not be
        caught by the generated ``FieldOperator`` instance.

    Returns:
        A backend-specific ``FieldOperator`` implementing the provided definition.

    """

    externals_defs = {**externals} if isinstance(externals, Mapping) else {}

    if callable(definition):
        ir = ff.FieldOperatorParser.apply_to_function(definition)
    else:
        if isinstance(definition, str):
            source_definition = ff.SourceDefinition(definition)
        elif isinstance(definition, tuple) and 2 <= len(definition) <= 3:
            source_definition = ff.SourceDefinition(*definition)
        else:
            raise common.GTValueError(f"Invalid field operator definition ({definition})")

        ir = ff.FieldOperatorParser.apply(
            source_definition, closure_vars=None, externals_defs=externals_defs
        )

    return backend.generate_operator(ir)
