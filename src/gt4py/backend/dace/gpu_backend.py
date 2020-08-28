import dace

from gt4py import backend as gt_backend

from .base_backend import (
    DaceBackend,
    CudaDaceOptimizer,
    DacePyModuleGenerator,
    dace_layout,
    dace_is_compatible_layout,
    dace_is_compatible_type,
)


class GPUDacePyModuleGenerator(DacePyModuleGenerator):
    def generate_field_ptr_str(self, arg):
        return f"{arg}.data.data.ptr"


class GPUDaceOptimizer(CudaDaceOptimizer):

    description = ""

    def transform_library(self, sdfg):
        from gt4py.backend.dace.sdfg.library.nodes import StencilLibraryNode

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, StencilLibraryNode):
                    node.loop_order = "JKI"

        from gt4py.backend.dace.sdfg.transforms import PruneTransientOutputs

        sdfg.apply_transformations_repeated(PruneTransientOutputs, validate=False)
        return sdfg

    def transform_optimize(self, sdfg):

        import dace

        from dace.transformation.dataflow import MapCollapse
        from gt4py.backend.dace.sdfg.transforms import OnTheFlyMapFusion
        from gt4py.backend.dace.sdfg.transforms import PrefetchingKCachesTransform

        sdfg.apply_transformations_repeated(MapCollapse, validate=False)
        sdfg.apply_strict_transformations(validate=False)

        for name, array in sdfg.arrays.items():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.Persistent

        sdfg.apply_transformations_repeated(OnTheFlyMapFusion, validate=False)


        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    kcache_subgraph = {
                        PrefetchingKCachesTransform._nsdfg_node: state.node_id(node)
                    }
                    trafo = PrefetchingKCachesTransform(
                        sdfg.sdfg_id, sdfg.node_id(state), kcache_subgraph, 0
                    )
                    trafo.storage_type = dace.dtypes.StorageType.Register
                    trafo.apply(sdfg)

        from dace.transformation.subgraph.subgraph_fusion import SubgraphFusion
        from dace.sdfg.graph import SubgraphView

        for graph in sdfg.nodes():
            subgraph = SubgraphView(
                graph, [node for node in graph.nodes() if graph.out_degree(node) > 0]
            )
            fusion = SubgraphFusion(subgraph)
            fusion.transient_allocation = dace.dtypes.StorageType.Register
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

        return sdfg


@gt_backend.register
class GPUDaceBackend(DaceBackend):
    name = "dacecuda"
    storage_info = {
        "alignment": 1,
        "device": "gpu",
        "layout_map": dace_layout,
        "is_compatible_layout": dace_is_compatible_layout,
        "is_compatible_type": dace_is_compatible_type,
    }
    GENERATOR_CLASS = GPUDacePyModuleGenerator
    DEFAULT_OPTIMIZER = GPUDaceOptimizer()
