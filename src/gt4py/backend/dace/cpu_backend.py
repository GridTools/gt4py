from gt4py import backend as gt_backend

from .base_backend import (
    DaceBackend,
    DacePyModuleGenerator,
    dace_layout,
    dace_is_compatible_layout,
    dace_is_compatible_type,
)


class CPUDacePyModuleGenerator(DacePyModuleGenerator):
    def generate_field_ptr_str(self, arg):
        return f"{arg}.ctypes.data"


@gt_backend.register
class CPUDaceBackend(DaceBackend):
    name = "dacex86"
    options = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": dace_layout,
        "is_compatible_layout": dace_is_compatible_layout,
        "is_compatible_type": dace_is_compatible_type,
    }
    GENERATOR_CLASS = CPUDacePyModuleGenerator

    @classmethod
    def transform_to_device(cls, sdfg):
        pass

    @classmethod
    def transform_library(cls, sdfg):
        from gt4py.backend.dace.sdfg.library.nodes import ApplyMethodLibraryNode

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, ApplyMethodLibraryNode):
                    node.loop_order = "IJK"

    @classmethod
    def transform_optimize(cls, sdfg):
        # from dace.transformation.dataflow.map_collapse import MapCollapse
        #
        # from gt4py.backend.dace.sdfg.transforms import global_ij_tiling
        #
        # sdfg.apply_transformations_repeated([MapCollapse], validate=False)
        # sdfg.apply_strict_transformations(validate=False)
        # global_ij_tiling(sdfg, tile_size=(8, 8))
        pass
