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
    def transform_optimize(cls, sdfg):
        # from dace.transformation.dataflow.map_expansion import MapExpansion
        # from dace.transformation.dataflow.map_collapse import MapCollapse
        #
        # from dace.transformation.dataflow import InLocalStorage, OutLocalStorage
        from dace.transformation.dataflow import MergeArrays

        # from dace.transformation.dataflow import MapFusion
        #
        # from dace.transformation.dataflow import MapToForLoop
        #
        # for state in sdfg.nodes():
        #     #     sdfg.apply_transformations(MapExpansion, states=[state], validate=False)
        #     #     sdfg.apply_transformations(MapCollapse, states=[state], validate=False)
        #     eliminate_trivial_k_loop(sdfg, state)
        # for state in sdfg.nodes():
        #     outer_k_loop_to_inner_map(sdfg, state)
        #
        from dace.transformation.interstate import StateFusion

        sdfg.apply_transformations_repeated(StateFusion)
        sdfg.apply_strict_transformations()
        # sdfg.apply_transformations_repeated([MapFusion], validate=False)
        # # from dace.transformation.dataflow import Vectorization
        #
        # #
        # # sdfg.apply_transformations_repeated(Vectorization)
        # from gt4py.backend.dace.sdfg.transforms import global_ij_tiling
        # global_ij_tiling(sdfg, tile_size=(8, 8))

    @classmethod
    def transform_2(cls, sdfg):
        from dace.transformation.dataflow import MapCollapse

        sdfg.apply_transformations_repeated(MapCollapse)
