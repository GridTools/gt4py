# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal, Optional, Union, overload

import numpy as np
import pytest

import dace
import copy
import uuid
from dace import data as dace_data, symbolic as dace_sym
from dace.sdfg import nodes as dace_nodes


def unique_name(name: str) -> str:
    """Adds a unique string to `name`.

    Note:
        The names generates by this function are rather unstable and it should
        not be used if a particular order should be enforced. This function is
        marked for deprecation.
    """
    maximal_length = 200
    unique_sufix = str(uuid.uuid1()).replace("-", "_")
    if len(name) > (maximal_length - len(unique_sufix)):
        name = name[: (maximal_length - len(unique_sufix) - 1)]
    return f"{name}_{unique_sufix}"


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

        sdfg_clone.name = unique_name(sdfg_clone.name)
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


def make_if_block(
    state: dace.SDFGState,
    b1_name: str = "__arg1",
    b2_name: str = "__arg2",
    cond_name: str = "__cond",
    output_name: str = "__output",
    b1_type: dace.typeclass = dace.float64,
    b2_type: dace.typeclass = dace.float64,
    output_type: dace.typeclass = dace.float64,
    b1_shape: Optional[tuple[int, ...]] = None,
    b2_shape: Optional[tuple[int, ...]] = None,
    output_shape: Optional[tuple[int, ...]] = None,
) -> dace_nodes.NestedSDFG:
    """Create an if block that selects between its arguments `b1_name` and `b2_name`.

    By default, the arguments and the output are scalars. If `b{1,2}_shape` or
    `output_shape` is given, the corresponding connector is instead connected to an
    array element `a[__i]` inside a Map with iteration variable `__i`. Because a nested
    SDFG connector has to be equivalent to the data it is connected to, the connector
    is then the full array, accessed at `[__i]`.
    """
    inner_sdfg = dace.SDFG(unique_name("if_stmt_"))

    types = {b1_name: b1_type, b2_name: b2_type, cond_name: dace.bool_, output_name: output_type}
    shapes = {b1_name: b1_shape, b2_name: b2_shape, output_name: output_shape}
    for name in {b1_name, b2_name, cond_name, output_name}:
        if shapes.get(name) is None:
            inner_sdfg.add_scalar(name, dtype=types[name], transient=False)
        else:
            inner_sdfg.add_array(name, shape=shapes[name], dtype=types[name], transient=False)
    symbol_mapping = None
    if any(shape is not None for shape in shapes.values()):
        inner_sdfg.add_symbol("__i", stype=dace.int32)
        symbol_mapping = {"__i": "__i"}
    b1_index = "0" if b1_shape is None else "__i"
    b2_index = "0" if b2_shape is None else "__i"
    output_index = "0" if output_shape is None else "__i"

    if_region = dace.sdfg.state.ConditionalBlock(unique_name("if"))
    inner_sdfg.add_node(if_region, is_start_block=True)

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=inner_sdfg)
    tstate = then_body.add_state("true_branch", is_start_block=True)
    tstate.add_nedge(
        tstate.add_access(b1_name),
        tstate.add_access(output_name),
        dace.Memlet(f"{b1_name}[{b1_index}] -> [{output_index}]"),
    )

    else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=inner_sdfg)
    fstate = else_body.add_state("false_branch", is_start_block=True)
    fstate.add_nedge(
        fstate.add_access(b2_name),
        fstate.add_access(output_name),
        dace.Memlet(f"{b2_name}[{b2_index}] -> [{output_index}]"),
    )

    if_region.add_branch(dace.sdfg.state.CodeBlock(cond_name), then_body)
    if_region.add_branch(dace.sdfg.state.CodeBlock(f"not {cond_name}"), else_body)

    return state.add_nested_sdfg(
        sdfg=inner_sdfg,
        inputs={b1_name, b2_name, cond_name},
        outputs={output_name},
        symbol_mapping=symbol_mapping,
    )


class DefinedSymbolsCounter:
    """Counts the calls made to ``dace.SDFGState.defined_symbols()``.

    Resolving the symbols of a state is an expensive operation that scans the whole
    SDFG, so it should only happen when a copied map scope actually contains a
    ``NestedSDFG`` and its result should be reused instead of being recomputed.
    This class patches ``defined_symbols()`` on the class, thus it counts the
    calls made on the states of every SDFG.

    Args:
        monkeypatch: The pytest monkey patch fixture used to install and
            automatically revert the patch.

    Note:
        The number of observed calls is available through the ``calls``
        attribute.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        original = dace.SDFGState.defined_symbols

        def counting(state: dace.SDFGState) -> dict[str, Any]:
            self.calls += 1
            return original(state)

        monkeypatch.setattr(dace.SDFGState, "defined_symbols", counting)
