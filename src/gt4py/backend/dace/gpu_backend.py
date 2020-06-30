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

        sdfg.apply_transformations(
            [GPUTransformSDFG], options={"strict_transform": False}, strict=False, validate=False
        )

        for st in sdfg.nodes():
            for node in st.nodes():
                parent = st.entry_node(node)
                if isinstance(node, dace.nodes.NestedSDFG) and (
                    parent is None or parent.schedule != dace.ScheduleType.GPU_Device
                ):
                    cls.transform_to_device(node.sdfg)

    @classmethod
    def transform_library(cls, sdfg):
        from gt4py.backend.dace.sdfg.library.nodes import ApplyMethodLibraryNode

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, ApplyMethodLibraryNode):
                    node.loop_order = "IKJ"

    @classmethod
    def transform_optimize(cls, sdfg):
        sdfg.apply_strict_transformations(validate=False)
        for name, array in sdfg.arrays.items():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.Persistent
        dace.Config.set("compiler", "cuda", "default_block_size", value="64,2,1")
