from gt4py import backend as gt_backend

from .base_backend import (
    DaceBackend,
    DaceOptimizer,
    DacePyModuleGenerator,
    dace_layout,
    dace_is_compatible_layout,
    dace_is_compatible_type,
)


class CPUDacePyModuleGenerator(DacePyModuleGenerator):
    def generate_field_ptr_str(self, arg):
        return f"{arg}.ctypes.data"


class X86DaceOptimizer(DaceOptimizer):

    description = "GT x86 style transformations "

    def transform_library(self, sdfg):
        from gt4py.backend.dace.sdfg.library.nodes import StencilLibraryNode

        from gt4py.backend.dace.sdfg.transforms import PruneTransientOutputs

        sdfg.apply_transformations_repeated(PruneTransientOutputs, validate=False)

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, StencilLibraryNode):
                    node.loop_order = "IJK"
        return sdfg

    def transform_optimize(self, sdfg):
        import dace
        from dace.transformation.dataflow import MapCollapse

        sdfg.apply_transformations_repeated(MapCollapse, validate=False)
        sdfg.apply_strict_transformations(validate=False)
        for name, array in sdfg.arrays.items():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.Persistent
        dace.Config.set("compiler", "cuda", "default_block_size", value="64,2,1")
        from daceperiments.transforms import OnTheFlyMapFusion

        sdfg.apply_transformations_repeated(OnTheFlyMapFusion, validate=False)

        # dace.Config.set("compiler", "cuda", "default_block_size", value="64,2,1")

        from dace.transformation.subgraph.subgraph_fusion import SubgraphFusion
        from dace.sdfg.graph import SubgraphView

        for graph in sdfg.nodes():
            subgraph = SubgraphView(
                graph, [node for node in graph.nodes() if graph.out_degree(node) > 0]
            )
            fusion = SubgraphFusion()
            fusion.apply(sdfg, subgraph)
        for name, array in sdfg.arrays.items():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.Scope
                array.storage = dace.dtypes.StorageType.CPU_Heap
        from gt4py.backend.dace.sdfg.transforms import PrefetchingKCachesTransform

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    node.sdfg.apply_transformations(
                        PrefetchingKCachesTransform,
                        options={"storage_type": dace.dtypes.StorageType.CPU_Heap},
                        validate=False,
                    )

        return sdfg
        #

    # def transform_optimize(self, sdfg):
    #     # from dace.transformation.dataflow.map_collapse import MapCollapse
    #     #
    #     # from gt4py.backend.dace.sdfg.transforms import global_ij_tiling
    #     #
    #     # sdfg.apply_transformations_repeated([MapCollapse], validate=False)
    #     # sdfg.apply_strict_transformations(validate=False)
    #     # global_ij_tiling(sdfg, tile_size=(8, 8))
    #     from dace.transformation.interstate import StateFusion
    #
    #     # sdfg.apply_transformations_repeated([StateFusion], strict=False, validate=False)
    #     from daceperiments.transforms import BasicRegisterCache
    #     from dace.transformation.dataflow import MapCollapse
    #     import dace.sdfg.utils
    #     from dace.transformation.subgraph.subgraph_fusion import SubgraphFusion
    #
    #     # for graph in sdfg.nodes():
    #     #     subgraph = dace.sdfg.graph.SubgraphView(
    #     #         graph, [node for node in graph.nodes() if graph.out_degree(node) > 0]
    #     #     )
    #     #     fusion = SubgraphFusion()
    #     #     fusion.apply(sdfg, subgraph)
    #
    #     # sdfg.apply_transformations_repeated(MapCollapse, validate=False)
    #     # for state in sdfg.nodes():
    #     #     for node in state.nodes():
    #     #         if isinstance(node, dace.nodes.NestedSDFG):
    #     #             for name, descr in node.sdfg.arrays.items():
    #     #                 if name == "data_col":
    #     #                     node.sdfg.apply_transformations(
    #     #                         BasicRegisterCache, options=dict(array=name), validate=False
    #     #                     )
    #     from dace.transformation.interstate import LoopPeeling
    #
    #     # for state in sdfg.nodes():
    #     #     for node in state.nodes():
    #     #         if isinstance(node, dace.nodes.NestedSDFG) and node.label.startswith("stencil_2"):
    #     #             node.sdfg.apply_transformations(
    #     #                 LoopPeeling, options=dict(count=1), validate=False
    #     #             )
    #     # for state in sdfg.nodes():
    #     #     for node in state.nodes():
    #     #         if isinstance(node, dace.nodes.NestedSDFG) and node.label.startswith("stencil_2"):
    #     #             node.sdfg.apply_transformations(
    #     #                 LoopPeeling, options=dict(count=1, begin=False), validate=False
    #     #             )
    #

    #     sdfg.validate()
    #     return sdfg


@gt_backend.register
class CPUDaceBackend(DaceBackend):
    name = "dacex86"
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": dace_layout,
        "is_compatible_layout": dace_is_compatible_layout,
        "is_compatible_type": dace_is_compatible_type,
    }
    GENERATOR_CLASS = CPUDacePyModuleGenerator
    DEFAULT_OPTIMIZER = X86DaceOptimizer()
