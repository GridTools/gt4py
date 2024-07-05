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

"""
Implementation of the Nabla4 Stencil.
"""

import copy

from gt4py.next.common import NeighborTable
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.ffront.fbuiltins import Field
from gt4py.next.program_processors.runners import dace_fieldview as dace_backend
from gt4py.next.type_system import type_specifications as ts
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    KDim,
    Cell,
    Edge,
    IDim,
    JDim,
    MeshDescriptor,
    V2EDim,
    Vertex,
    simple_mesh,
    skip_value_mesh,
)
from typing import Sequence, Any
from functools import reduce
import numpy as np

import dace

wpfloat = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
SIZE_TYPE = ts.ScalarType(ts.ScalarKind.INT32)


def nabla4_np(
    N: Field[[Edge, KDim], wpfloat],
    T: Field[[Edge, KDim], wpfloat],
    **kwargs,  # Allows to use the same call argument object as for the SDFG
) -> Field[[Edge, KDim], wpfloat]:
    return N + T


def dace_strides(
    array: np.ndarray,
    name: None | str = None,
) -> tuple[int, ...] | dict[str, int]:
    if not hasattr(array, "strides"):
        return {}
    strides = array.strides
    if hasattr(array, "itemsize"):
        strides = tuple(stride // array.itemsize for stride in strides)
    if name is not None:
        strides = {f"__{name}_stride_{i}": stride for i, stride in enumerate(strides)}
    return strides


def dace_shape(
    array: np.ndarray,
    name: str,
) -> dict[str, int]:
    if not hasattr(array, "shape"):
        return {}
    return {f"__{name}_size_{i}": size for i, size in enumerate(array.shape)}


def make_syms(**kwargs: np.ndarray) -> dict[str, int]:
    SYMBS = {}
    for name, array in kwargs.items():
        SYMBS.update(**dace_shape(array, name))
        SYMBS.update(**dace_strides(array, name))
    return SYMBS


def build_nambla4_gtir():
    edge_k_domain = im.call("unstructured_domain")(
        im.call("named_range")(itir.AxisLiteral(value=Edge.value, kind=Edge.kind), 0, "num_edges"),
        im.call("named_range")(
            itir.AxisLiteral(value=KDim.value, kind=KDim.kind), 0, "num_k_levels"
        ),
    )

    num_edges = 27
    num_k_levels = 10

    EK_FTYPE = ts.FieldType(dims=[Edge, KDim], dtype=wpfloat)

    nabla4prog = itir.Program(
        id="nabla4_partial",
        function_definitions=[],
        params=[
            itir.Sym(id="T", type=EK_FTYPE),
            itir.Sym(id="N", type=EK_FTYPE),
            itir.Sym(id="nab4", type=EK_FTYPE),
            itir.Sym(id="num_edges", type=SIZE_TYPE),
            itir.Sym(id="num_k_levels", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("N_it", "T_it")(im.plus(im.deref("T_it"), im.deref("N_it"))),
                        edge_k_domain,
                    )
                )("N", "T"),
                domain=edge_k_domain,
                target=itir.SymRef(id="nab4"),
            )
        ],
    )

    offset_provider = {}

    N = np.random.rand(num_edges, num_k_levels)
    T = np.random.rand(num_edges, num_k_levels)
    nab4 = np.empty_like(N)

    sdfg = dace_backend.build_sdfg_from_gtir(nabla4prog, offset_provider)

    call_args = dict(
        T=T,
        N=N,
        nab4=nab4,
        num_edges=num_edges,
        num_k_levels=num_k_levels,
    )
    SYMBS = make_syms(**call_args)

    sdfg(**call_args, **SYMBS)
    ref = nabla4_np(**call_args)

    assert np.allclose(ref, nab4)
    print(f"Test succeeded")


if "__main__" == __name__:
    build_nambla4_gtir()

    """
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("NpT")(
                            im.multiplies_(im.deref("NpT"), 4.0)
                        ),
                        edge_k_domain,
                    )
                )(
                    im.call(
                        im.call("as_fieldop")(
                            im.lambda_("N", "T")(
                                im.plus(im.deref("N"), im.deref("T"))
                            ),
                            edge_k_domain,
                        )
                    )("N", "T")
                ),
                domain=edge_k_domain,
                target=itir.SymRef(id="nab4"),
            )
        ],
        """
