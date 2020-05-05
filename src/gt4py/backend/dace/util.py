import ctypes

import dace.subsets

from gt4py import ir as gt_ir


def axis_interval_to_range(interval: gt_ir.AxisInterval):
    k_start = (
        interval.start.offset
        if interval.start.level == gt_ir.LevelMarker.START
        else "K{:+d}".format(interval.start.offset)
    )
    k_end = (
        interval.end.offset
        if interval.end.level == gt_ir.LevelMarker.START
        else "K{:+d}".format(interval.end.offset)
    )
    return dace.subsets.Range.from_string(f"{k_start}:{k_end}")


def load_dace_program(dace_ext_lib):
    #    if dace_ext_lib in LOADED_REGISTRY:
    #        return LOADED_REGISTRY[dace_ext_lib]
    #    else:
    #    dll = ReloadableDLL(dace_ext_lib, module_name)
    #    sdfg = SDFG.from_file(dace_build_path + "/program.sdfg")
    #    compiled_sdfg = CompiledSDFG(sdfg, dll)
    #    LOADED_REGISTRY[dace_ext_lib] = compiled_sdfg
    #    return compiled_sdfg
    dace_lib = ctypes.CDLL(dace_ext_lib)
    return dace_lib
