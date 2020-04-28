import ctypes


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
