import dace

from gt4py import backend as gt_backend

from .base_backend import (
    DaceBackend,
    DacePyModuleGenerator,
    dace_layout,
    dace_is_compatible_layout,
    dace_is_compatible_type,
)


class GPUDacePyModuleGenerator(DacePyModuleGenerator):
    def generate_field_ptr_str(self, arg):
        return f"{arg}.data.data.ptr"


@gt_backend.register
class GPUDaceBackend(DaceBackend):
    name = "dacecuda"
    options = {}
    storage_info = {
        "alignment": 1,
        "device": "gpu",
        "layout_map": dace_layout,
        "is_compatible_layout": dace_is_compatible_layout,
        "is_compatible_type": dace_is_compatible_type,
    }
    GENERATOR_CLASS = GPUDacePyModuleGenerator

    @classmethod
    def transform_to_device(cls, sdfg):
        for name, array in sdfg.arrays.items():
            array.storage = dace.dtypes.StorageType.GPU_Global
        from dace.transformation.interstate.gpu_transform_sdfg import GPUTransformSDFG

        sdfg: dace.SDFG
        sdfg.apply_transformations([GPUTransformSDFG], strict=False, validate=False)

        for st in sdfg.nodes():
            for node in st.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    cls.transform_to_device(node.sdfg)

    @classmethod
    def transform_optimize(cls, sdfg):
        for name, array in sdfg.arrays.items():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.Persistent
        dace.Config.set("compiler", "cuda", "default_block_size", value="64,2,1")

        # tiling transform:
        # from dace.transformation.dataflow.tiling import MapTiling
        # from dace.transformation.dataflow import InLocalStorage, OutLocalStorage
        #
        # for state in sdfg.nodes():
        #     sdfg.apply_transformations(MapTiling, states=[state], validate=False)
        # # for state in sdfg.nodes():
        # #     sdfg.apply_transformations(
        # #         MapTiling, options=dict(tile_sizes=(64, 64, 8)), states=[state], validate=False
        # #     )
        # #     sdfg.apply_transformations(
        # #         MapTiling, options=dict(tile_sizes=(8, 8, 1),), states=[state], validate=False
        # #     )
        #
        # sdfg.apply_strict_transformations(validate=False)
        # sdfg.apply_transformations_repeated([InLocalStorage, OutLocalStorage], validate=False)
