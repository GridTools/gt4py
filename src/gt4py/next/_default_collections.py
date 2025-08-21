# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.next import containers
from gt4py.next.ffront import type_specifications as ts_ffront
from gt4py.next.type_system import type_specifications as ts, type_translation


def _type_of_dataclass(cls: type) -> ts_ffront.NamedTupleType:
    assert dataclasses.is_dataclass(cls)
    fields = dataclasses.fields(cls)

    types = [type_translation.from_type_hint(field.type) for field in fields]
    fields = [field for field in dataclasses.fields(cls)]
    keys = [f.name for f in fields]
    return_type = ts_ffront.NamedTupleType(types=types, keys=keys)

    return ts_ffront.ConstructorType(
        definition=ts.FunctionType(
            pos_or_kw_args=dict(zip(keys, types, strict=True)),
            kw_only_args={},
            pos_only_args=[],
            returns=return_type,
        )
    )


containers.register(lambda cls: dataclasses.is_dataclass(cls), _type_of_dataclass)
