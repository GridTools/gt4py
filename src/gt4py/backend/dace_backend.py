import os
import numpy as np

from gt4py import backend as gt_backend
from gt4py.utils import text as gt_text
import dace
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL
from dace.sdfg import SDFG

LOADED_REGISTRY = dict()


def dace_layout(mask):
    ctr = iter(range(sum(mask)))
    layout = [next(ctr) if m else None for m in mask]
    return tuple(layout)


def dace_is_compatible_layout(field):
    return sum(field.shape) > 0


def dace_is_compatible_type(field):
    return isinstance(field, np.ndarray)


def load_dace_program(dace_build_path, dace_ext_lib, module_name):
    if dace_ext_lib in LOADED_REGISTRY:
        return LOADED_REGISTRY[dace_ext_lib]
    else:
        dll = ReloadableDLL(dace_ext_lib, module_name)
        sdfg = SDFG.from_file(dace_build_path + "/program.sdfg")
        compiled_sdfg = CompiledSDFG(sdfg, dll)
        LOADED_REGISTRY[dace_ext_lib] = compiled_sdfg
        return compiled_sdfg


class PythonDaceGenerator(gt_backend.BaseModuleGenerator):
    def __init__(self, backend_class, options):
        super().__init__(backend_class, options)

    def generate_imports(self):
        source = f"""
import functools
from gt4py.backend.dace_backend import load_dace_program

dace_program = load_dace_program("{self.options.dace_build_path}", "{self.options.dace_ext_lib}", "{self.options.dace_module_name}")
# dace_program = load_dace_program("{self.options.dace_ext_lib}", "{self.options.name}")
        """
        return source

    def generate_implementation(self):
        sources = gt_text.TextBlock(indent_size=gt_backend.BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        for arg in self.implementation_ir.api_signature:
            if True or arg.name not in self.implementation_ir.unreferenced:
                args.append(arg.name)
                # if arg.name in self.implementation_ir.fields:
                #     args.append("list(_origin_['{}'])".format(arg.name))

        source = """
dace_program({run_args}, I=np.int32(_domain_[0]), J=np.int32(_domain_[1]), K=np.int32(_domain_[2]))
""".format(
            run_args=", ".join([f"{n}={n}" for n in args])
        )

        source = source + (
            """if exec_info is not None:
    exec_info["run_end_time"] = time.perf_counter()
"""
        )
        sources.extend(source.splitlines())

        return sources.text


@gt_backend.register
class DaceBackend(gt_backend.BaseBackend):
    name = "dace"
    options = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": dace_layout,
        "is_compatible_layout": dace_is_compatible_layout,
        "is_compatible_type": dace_is_compatible_type,
    }

    GENERATOR_CLASS = PythonDaceGenerator

    @classmethod
    def get_dace_module_path(cls, stencil_id):
        path = os.path.join(
            cls.get_stencil_package_path(stencil_id),
            cls.get_stencil_module_name(stencil_id, qualified=False),
        )

        return path

    @classmethod
    def generate_dace(cls, stencil_id, implementation_ir, options):
        sdfg = implementation_ir.dace_program.to_sdfg()
        sdfg.apply_strict_transformations()

        dace_build_path = os.path.relpath(cls.get_dace_module_path(stencil_id))
        os.makedirs(dace_build_path, exist_ok=True)

        program_folder = dace.codegen.compiler.generate_program_folder(
            sdfg=sdfg,
            code_objects=dace.codegen.codegen.generate_code(sdfg),
            out_path=dace_build_path,
        )
        assert program_folder == dace_build_path
        dace_ext_lib = dace.codegen.compiler.configure_and_compile(program_folder)

        return dace_ext_lib, dace_build_path

    @classmethod
    def build(cls, stencil_id, implementation_ir, definition_func, options):
        cls._check_options(options)

        # Generate the Python binary extension with dace
        dace_ext_lib, dace_build_path = cls.generate_dace(stencil_id, implementation_ir, options)

        # Generate and return the Python wrapper class
        generator_options = options.as_dict()
        generator_options["dace_ext_lib"] = dace_ext_lib
        generator_options["dace_build_path"] = dace_build_path
        generator_options["dace_module_name"] = cls.get_stencil_module_name(
            stencil_id=stencil_id, qualified=False
        )

        extra_cache_info = {"dace_build_path": dace_build_path}

        return super(DaceBackend, cls)._build(
            stencil_id, implementation_ir, definition_func, generator_options, extra_cache_info
        )
