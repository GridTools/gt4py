# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal, Union, overload, Any

import numpy as np
import dace
import copy
from dace import data as dace_data, symbolic as dace_sym
from dace.sdfg import nodes as dace_nodes
from gt4py.next.program_processors.runners.dace.transformations import (
    utils as gtx_transformations_utils,
)


@overload
def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[False],
) -> int: ...


@overload
def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[True],
) -> list[dace_nodes.Node]: ...


def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: bool = False,
) -> Union[int, list[dace_nodes.Node]]:
    """Counts the number of nodes in of a particular type in `graph`.

    If `graph` is an SDFGState then only count the nodes inside this state,
    but if `graph` is an SDFG count in all states.

    Args:
        graph: The graph to scan.
        node_type: The type or sequence of types of nodes to look for.
    """

    states = graph.states() if isinstance(graph, dace.SDFG) else [graph]
    found_nodes: list[dace_nodes.Node] = []
    for state_nodes in states:
        for node in state_nodes.nodes():
            if isinstance(node, node_type):
                found_nodes.append(node)
    if return_nodes:
        return found_nodes
    return len(found_nodes)


def compile_and_run_sdfg(
    sdfg: dace.SDFG,
    *args: Any,
    **kwargs: Any,
) -> dace.codegen.CompiledSDFG:
    """This function guarantees that the SDFG is compiled and run.

    This function will modify the name of the SDFG to ensure that the code is
    regenerated and recompiled properly. It will also suppress warnings about
    shared objects that are loaded multiple times.
    """

    with dace.config.set_temporary("compiler.use_cache", value=False):
        sdfg_clone = copy.deepcopy(sdfg)

        sdfg_clone.name = gtx_transformations_utils.unique_name(sdfg_clone.name)
        sdfg_clone._recompile = True
        sdfg_clone._regenerate_code = True  # TODO(phimuell): Find out if it has an effect.
        csdfg = sdfg_clone.compile()
        csdfg(*args, **kwargs)

    return csdfg


def make_sdfg_args(
    sdfg: dace.SDFG,
    symbols: dict[str, Any] = {},
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generates the arguments to call the SDFG.

    The function returns data for the reference and the result call.
    You can compare the two using `compar_sdfg_res()`.
    """
    sdfg_args = sdfg.signature_arglist(False)
    rng = np.random.default_rng()

    try:
        import cupy

        def cp_ndarray(buffer, **kwargs):
            assert "dtype" not in kwargs
            assert buffer.ndim == 1
            return cupy.ndarray(memptr=cupy.array(buffer, copy=True), dtype=buffer.dtype, **kwargs)

    except (ImportError, ModuleNotFoundError):

        def cp_ndarray(buffer, **kwargs):
            raise ModuleNotFoundError("Requested GPU memory, but no GPU found.")

    def np_ndarray(buffer, **kwargs):
        assert "dtype" not in kwargs
        assert buffer.ndim == 1
        return np.ndarray(buffer=np.array(buffer, copy=True), dtype=buffer.dtype, **kwargs)

    def host_rand(size, dtype):
        np_dtype = dtype.as_numpy_dtype()
        if dtype is dace.bool_:
            return rng.random(size=int(size), dtype=np.float32) > 0.5

        elif dtype in dace.dtypes.INTEGER_TYPES:
            np_limits = np.iinfo(np_dtype)
            return rng.integers(
                low=np_limits.min,
                high=np_limits.max,
                size=int(size),
                dtype=np_dtype,
            )
        return rng.random(size=int(size), dtype=np_dtype)

    # We first have to generate the symbols, because they might appear in the shape.
    used_symbols: dict[str, Any] = {}
    for arg_name in sdfg_args + list(symbols.keys()):
        if arg_name not in sdfg.symbols:
            continue
        arg_type = sdfg.symbols[arg_name]
        if arg_name in symbols:
            used_symbols[arg_name] = arg_type(symbols[arg_name])
        else:
            used_symbols[arg_name] = host_rand(1, dtype=sdfg.symbols[arg_name])[0]

    # Now generate the non array arguments.
    ref: dict[str, Any] = {}
    res: dict[str, Any] = {}
    for arg_name in sdfg_args:
        if arg_name in sdfg.arrays:
            desc = sdfg.arrays[arg_name]
            assert not desc.transient

            if isinstance(desc, dace.data.Scalar):
                ref[arg_name] = host_rand(1, desc.dtype)[0]
                res[arg_name] = copy.deepcopy(ref[arg_name])
                continue

            ndarray = (
                cp_ndarray if desc.storage is dace.dtypes.StorageType.GPU_Global else np_ndarray
            )
            host_buffer = host_rand(dace_sym.evaluate(desc.total_size, used_symbols), desc.dtype)
            shape = tuple(dace_sym.evaluate(s, used_symbols) for s in desc.shape)
            dtype = desc.dtype.as_numpy_dtype()
            strides = tuple(
                dace_sym.evaluate(s, used_symbols) * desc.dtype.bytes for s in desc.strides
            )

            # We have to do it this way to ensure that the strides are copied correctly.
            ref[arg_name] = ndarray(shape=shape, buffer=host_buffer, strides=strides)
            res[arg_name] = ndarray(shape=shape, buffer=host_buffer, strides=strides)

        elif arg_name in used_symbols:
            ref[arg_name] = used_symbols[arg_name]
            res[arg_name] = copy.deepcopy(ref[arg_name])

        else:
            raise ValueError(f"Could not find argument: {arg_name}")

    return ref, res


def compare_sdfg_res(
    ref: dict[str, Any],
    res: dict[str, Any],
) -> bool:
    """Compares if `res` and  `ref` are the same."""
    return all(np.allclose(ref[name], res[name]) for name in ref.keys())
