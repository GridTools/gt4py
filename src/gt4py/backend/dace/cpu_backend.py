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
        # from dace.transformation.dataflow.map_collapse import MapCollapse
        #
        # from gt4py.backend.dace.sdfg.transforms import global_ij_tiling
        #
        # sdfg.apply_transformations_repeated([MapCollapse], validate=False)
        # sdfg.apply_strict_transformations(validate=False)
        # global_ij_tiling(sdfg, tile_size=(8, 8))
        from dace.transformation.interstate import StateFusion
        # sdfg.apply_transformations_repeated([StateFusion], strict=False, validate=False)
        return sdfg


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
