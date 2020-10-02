# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import gt4py.utils as gt_utils
import collections
import numpy as np
import types
import inspect

import dace.codegen.instrumentation


def copy_func(f, name=None):
    return types.FunctionType(
        f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__
    )


def annotate_function(function, dtypes):
    annotated_func = copy_func(function)
    for k in inspect.getfullargspec(annotated_func).args:
        annotated_func.__annotations__[k] = dtypes[k]
    for k in inspect.getfullargspec(annotated_func).kwonlyargs:
        annotated_func.__annotations__[k] = dtypes[k]
    return annotated_func


def standardize_dtype_dict(dtypes):
    """Standardizes the dtype dict as it can be specified for the stencil test suites.
    In the input dictionary, a selection of possible dtypes or just a single dtype can be specified for a set of fields
    or a single field. This function makes sure that all keys are tuples (by wrapping single field names and single
    dtypes as 1-tuples)"""
    assert isinstance(dtypes, collections.abc.Mapping)
    assert all(
        (isinstance(k, str) or gt_utils.is_iterable_of(k, str)) for k in dtypes.keys()
    ), "Invalid key in 'dtypes'."
    assert all(
        (isinstance(k, type) or gt_utils.is_iterable_of(k, type)) for k in dtypes.values()
    ), "Invalid dtype in 'dtypes'"

    result = {}
    for key, value in dtypes.items():
        if isinstance(key, str):
            key = (key,)
        else:
            key = (*key,)
        if isinstance(value, type):
            value = (value,)
        else:
            value = (*value,)
        result[key] = value

    for key, value in result.items():
        result[key] = [np.dtype(dt) for dt in result[key]]

    keys = [k for t in result.keys() for k in t]
    if not len(keys) == len(set(keys)):
        raise ValueError("Any field can be in only one group.")
    return result


import gt4py
from gt4py.backend.dace.base_backend import DaceOptimizer


class ApplyOTFOptimizer(DaceOptimizer):
    def transform_optimize(self, sdfg):
        from gt4py.backend.dace.sdfg.transforms import OnTheFlyMapFusion

        sdfg.apply_transformations_repeated(OnTheFlyMapFusion, validate=False)
        return sdfg


class DeduplicateAccesses(DaceOptimizer):
    def transform_optimize(self, sdfg):
        from dace.transformation.dataflow.dedup_access import DeduplicateAccess

        sdfg.apply_transformations_repeated(DeduplicateAccess, validate=False)
        return sdfg


class SubgraphFusion(DaceOptimizer):
    def __init__(self, storage_type: dace.dtypes.StorageType = dace.dtypes.StorageType.Register):
        self.storage_type = storage_type

    def transform_optimize(self, sdfg):
        from dace.transformation.subgraph.subgraph_fusion import SubgraphFusion
        from dace.sdfg.graph import SubgraphView

        for graph in sdfg.nodes():
            subgraph = SubgraphView(
                graph, [node for node in graph.nodes() if graph.out_degree(node) > 0]
            )
            fusion = SubgraphFusion(subgraph)
            fusion.transient_allocation = self.storage_type
            fusion.apply(sdfg)
            for name, array in sdfg.arrays.items():
                if array.transient:
                    if array.storage == dace.dtypes.StorageType.GPU_Global:
                        array.lifetime = dace.dtypes.AllocationLifetime.Persistent

                    for node in graph.nodes():
                        if isinstance(node, dace.nodes.NestedSDFG):
                            for inner_name, inner_array in node.sdfg.arrays.items():
                                if inner_name == name:
                                    inner_array.storage = array.storage
                                    inner_array.strides = array.strides

        dace.sdfg.utils.consolidate_edges(sdfg)
        return sdfg


class TileMap(DaceOptimizer):
    def __init__(self, dimension, tile_size, divides_evenly=False):
        self.dimension = dimension.lower()
        self.tile_size = tile_size
        self.divides_evenly = divides_evenly

    def transform_optimize(self, sdfg: dace.SDFG):

        from dace.transformation.dataflow import (
            MapCollapse,
            MapExpansion,
            MapTiling,
            MapInterchange,
        )

        sdfg.apply_transformations_repeated(MapExpansion, validate=False)
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and self.dimension in node.map.params:
                    candidate = {MapTiling._map_entry: state.node_id(node)}
                    trafo = MapTiling(sdfg.sdfg_id, sdfg.node_id(state), candidate, 0)
                    trafo.tile_sizes = (self.tile_size,)
                    trafo.divides_evenly = self.divides_evenly
                    trafo.apply(sdfg)
        # bubble sort the maps :D
        applied = True
        while applied:
            applied = False
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.MapEntry) and self.dimension in node.map.params:
                        for other_node in (edge.dst for edge in state.out_edges(node)):
                            if isinstance(other_node, dace.nodes.MapEntry):
                                candidate = {
                                    MapInterchange._outer_map_entry: state.node_id(node),
                                    MapInterchange._inner_map_entry: state.node_id(other_node),
                                }
                                trafo = MapInterchange(
                                    sdfg.sdfg_id, sdfg.node_id(state), candidate, 0
                                )
                                trafo.apply(sdfg)
                                from dace.sdfg.propagation import propagate_memlets_sdfg

                                applied = True
                                break
        sdfg.apply_transformations_repeated(MapCollapse, validate=False)

        return sdfg


class MapToFor(DaceOptimizer):
    def __init__(self, dimension):
        self.dimension = dimension.lower()

    def transform_optimize(self, sdfg):
        from dace.transformation.dataflow import MapExpansion, MapCollapse, MapToForLoop

        sdfg.apply_transformations_repeated(MapExpansion, validate=False)
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and self.dimension in node.map.params:
                    candidate = {MapToForLoop._map_entry: state.node_id(node)}
                    trafo = MapToForLoop(sdfg.sdfg_id, sdfg.node_id(state), candidate, 0)
                    trafo.apply(sdfg)
        sdfg.apply_transformations_repeated(MapCollapse, validate=False)

        return sdfg


class PrefetchingKCaches(DaceOptimizer):
    def __init__(self, arrays=None, storage_type=dace.dtypes.StorageType.Register):
        self.storage_type = storage_type
        self.arrays = arrays

    def transform_optimize(self, sdfg):
        from gt4py.backend.dace.sdfg.transforms import PrefetchingKCachesTransform

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    kcache_subgraph = {
                        PrefetchingKCachesTransform._nsdfg_node: state.node_id(node)
                    }
                    trafo = PrefetchingKCachesTransform(
                        sdfg.sdfg_id, sdfg.node_id(state), kcache_subgraph, 0
                    )
                    trafo.arrays = self.arrays
                    trafo.storage_type = self.storage_type
                    trafo.apply(sdfg)
        return sdfg


class PruneTransientOutputs(DaceOptimizer):
    def transform_library(self, sdfg):
        from gt4py.backend.dace.sdfg.transforms import PruneTransientOutputs

        sdfg.apply_transformations_repeated(PruneTransientOutputs, validate=False)
        return sdfg


class SpecializeIJK(DaceOptimizer):
    def __init__(self, domain, strides):
        self.domain = domain
        self.strides = strides

    def transform_optimize(self, sdfg):
        specialize_symbols = dict()
        specialize_symbols["I"] = self.domain[0]
        specialize_symbols["J"] = self.domain[1]
        specialize_symbols["K"] = self.domain[2]
        for name in sdfg.arrays:
            for var, stride in zip("IJK", self.strides):
                specialize_symbols[f"_{name}_{var}_stride"] = stride
        sdfg.specialize(specialize_symbols)
        return sdfg


def build_dace_adhoc(
    definition,
    domain,
    halo,
    specialize_strides,
    dtype,
    passes,
    alignment,
    layout,
    loop_order,
    device,
    constants=None,
    **params,
) -> gt4py.stencil_object.StencilObject:
    backend_name = f"dace_adhoc_{device}_{dtype}_{loop_order}_{alignment}_"
    backend_name += "_".join(str(int(h)) for h in halo) + "_"
    backend_name += "_".join(str(int(d)) for d in domain) + "_"
    backend_name += "_".join(str(int(s)) for s in specialize_strides) + "_"
    backend_name += "_".join(type(p).__name__ for p in passes)
    if len(params) > 0:
        backend_name += "_" + "_".join(f"{k}_{v}" for k, v in params.items())

    from gt4py.backend.dace.cpu_backend import CPUDaceBackend
    from gt4py.backend.dace.gpu_backend import GPUDaceBackend
    from gt4py.backend.dace.base_backend import CudaDaceOptimizer, CPUDaceOptimizer
    from gt4py.backend.concepts import register as register_backend

    base_backend = CPUDaceBackend if device == "cpu" else GPUDaceBackend
    base_optimizer = CPUDaceOptimizer if device == "cpu" else CudaDaceOptimizer

    backend_opts = {}
    if device == "gpu" and "gpu_block_size" in params:
        backend_opts["gpu_block_size"] = params["gpu_block_size"]
    if "specialize_sdfg_vars" in params:
        backend_opts["specialize_sdfg_vars"] = params["specialize_sdfg_vars"]
    backend_opts["computation_layout"] = loop_order
    backend_opts["enforce_dtype"] = dtype

    class CompositeOptimizer(base_optimizer):
        def __init__(self, passes):
            self._passes = [SpecializeIJK(domain, specialize_strides)] + passes

        def transform_to_device(self, sdfg):
            sdfg = super().transform_to_device(sdfg)
            return sdfg

        def transform_library(self, sdfg):
            for xform in self._passes:
                sdfg = xform.transform_library(sdfg)
            return sdfg

        def transform_optimize(self, sdfg):
            from dace.transformation.dataflow import MapCollapse

            sdfg.apply_transformations_repeated(MapCollapse, validate=False)
            for xform in self._passes:
                sdfg = xform.transform_optimize(sdfg)
            return sdfg

    @register_backend
    class AdHocBackend(base_backend):
        name = backend_name
        storage_info = {
            "alignment": alignment,  # will not affect temporaries currently
            "device": device,  # change me
            "layout_map": lambda m: layout,
            "is_compatible_layout": lambda m: True,
            "is_compatible_type": lambda m: True,
        }
        DEFAULT_OPTIMIZER = CompositeOptimizer(passes)

    constants = constants or {}
    return gt4py.gtscript.stencil(
        definition=definition,
        backend=backend_name,
        dtypes={"dtype": dtype},
        externals=constants,
        save_intermediate=True,
        **backend_opts,
    )
