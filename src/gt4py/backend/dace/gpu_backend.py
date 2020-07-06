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
                    node.loop_order = "IJK"

        from gt4py.backend.dace.sdfg.transforms import PruneTransientOutputs

        sdfg.apply_transformations_repeated(PruneTransientOutputs, validate=False)
        return sdfg

    def transform_optimize(self, sdfg):
        sdfg.apply_strict_transformations(validate=False)
        for name, array in sdfg.arrays.items():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.Persistent
        dace.Config.set("compiler", "cuda", "default_block_size", value="64,2,1")
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
