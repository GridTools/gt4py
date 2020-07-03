import os
import copy

import numpy as np

import dace
from dace.codegen.codegen import generate_code
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL

import gt4py.analysis as gt_analysis
from gt4py import backend as gt_backend
from gt4py.ir import DataType
from gt4py.utils import text as gt_text

from .sdfg.builder import SDFGBuilder


def dace_layout(mask):
    ctr = iter(range(sum(mask)))
    layout = [next(ctr) if m else None for m in mask]
    return tuple(layout)


def dace_is_compatible_layout(field):
    return sum(field.shape) > 0


def dace_is_compatible_type(field):
    return isinstance(field, np.ndarray)


class DacePyModuleGenerator(gt_backend.BaseModuleGenerator):
    def __init__(self, backend_class, options):
        super().__init__(backend_class, options)

    def generate_imports(self):
        source = f"""
import functools
import ctypes

from gt4py.backend.dace.util import load_dace_program

dace_lib = load_dace_program("{self.options.dace_ext_lib}")
"""
        return source

    def generate_implementation(self):
        sources = gt_text.TextBlock(
            indent_size=gt_backend.BaseModuleGenerator.TEMPLATE_INDENT_SIZE
        )

        args = []
        for arg in self.implementation_ir.api_signature:
            if True or arg.name not in self.implementation_ir.unreferenced:
                args.append(arg.name)
                # if arg.name in self.implementation_ir.fields:
                #     args.append("list(_origin_['{}'])".format(arg.name))
        field_slices = []

        for field_name in self.implementation_ir.arg_fields:
            if field_name not in self.implementation_ir.unreferenced:
                field_slices.append(
                    """#{name}_copy = np.array({name}.view(np.ndarray)[{slice}], copy=True) if {name} is not None else None
#_{name}_I, _{name}_J, _{name}_K = {name}_copy.shape
#if {name} is not None:
#    assert {name}_copy.shape=={shape}, str({name}_copy.shape)+'!='+str({shape})
#    assert {name}_copy.strides==tuple({name}_copy.itemsize*_gt_s_ for _gt_s_ in {strides}), str({name}_copy.strides)+'!='+str(tuple({name}.itemsize*_gt_s_ for _gt_s_ in {strides}))
{name} = {name}[{slice}] if {name} is not None else None
#if {name} is not None:
#   assert {name}.shape=={shape}, str({name}.shape)+'!='+str({shape})
#   assert {name}.strides==tuple({name}.itemsize*_gt_s_ for _gt_s_ in {strides}), str({name}.strides)+'!='+str(tuple({name}.itemsize*_gt_s_ for _gt_s_ in {strides}))""".format(
                        name=field_name,
                        slice=",".join(
                            f'_origin_["{field_name}"][{i}] - {-self.implementation_ir.fields_extents[field_name][i][0]}:_origin_["{field_name}"][{i}] - {-self.implementation_ir.fields_extents[field_name][i][0]}+_domain_[{i}] + {self.implementation_ir.fields_extents[field_name].frame_size[i]}'
                            if self.implementation_ir.fields[field_name].axes[i]
                            != self.implementation_ir.domain.sequential_axis.name
                            else f'_origin_["{field_name}"][{i}]:_origin_["{field_name}"][{i}]+_domain_[{i}]'
                            # else ":"
                            for i in range(len(self.implementation_ir.fields[field_name].axes))
                        ),
                        shape=tuple(
                            s
                            if self.implementation_ir.fields[field_name].axes[i]
                            != self.implementation_ir.domain.sequential_axis.name
                            else dace.symbol("K")
                            for i, s in enumerate(
                                self.implementation_ir.sdfg.arrays[field_name].shape
                            )
                        ),
                        strides=self.implementation_ir.sdfg.arrays[field_name].strides,
                    )
                )
        total_field_sizes = []
        symbol_ctype_strs = dict(I="ctypes.c_int(I)", J="ctypes.c_int(J)", K="ctypes.c_int(K)")
        for field_name in self.implementation_ir.arg_fields:
            if field_name not in self.implementation_ir.unreferenced:
                total_field_sizes.append(
                    f"_{field_name}_K = np.int32({field_name}.strides[1])/{field_name}.itemsize if not {field_name} is None else 0.0"
                )
                symbol_ctype_strs[f"_{field_name}_K"] = f"ctypes.c_int(np.int32(_{field_name}_K))"
                total_field_sizes.append(
                    f"_{field_name}_J = np.int32({field_name}.strides[0]/_{field_name}_K)/{field_name}.itemsize if not {field_name} is None else 0.0"
                )
                symbol_ctype_strs[f"_{field_name}_J"] = f"ctypes.c_int(np.int32(_{field_name}_J))"
                total_field_sizes.append(
                    # f"_{field_name}_I = np.int32({field_name}.strides[0]/_{field_name}_K)/{field_name}.itemsize if not {field_name} is None else 0.0"
                    f"_{field_name}_I = np.int32({field_name}.shape[0]) if not {field_name} is None else 0.0"
                )
        total_field_sizes = "\n".join(total_field_sizes)
        run_args_names = sorted(args)
        run_args_strs = []
        for name, datadescr in self.implementation_ir.sdfg.arglist().items():
            if name in self.implementation_ir.arg_fields:
                run_args_strs.append(f"ctypes.c_void_p({self.generate_field_ptr_str(name)})")
            else:
                run_args_strs.append(
                    # "np.ctypeslib.as_ctypes(np.{dtype}({par_name}))".format(
                    "ctypes.{ctype_name}(np.{numpy_type}({par_name}))".format(
                        ctype_name=DataType.from_dtype(datadescr.dtype.type).ctypes_str,
                        numpy_type=datadescr.dtype.type.__name__,
                        par_name=name,
                    )
                )

        # for str in []:
        #             field_run_args=", ".join(
        #                 [
        #                     f"ctypes.c_void_p({n}.ctypes.data)"
        #                     # f"np.ctypeslib.as_ctypes({n})"
        #                     for n in args
        #                     if n in self.implementation_ir.arg_fields
        #                 ]
        #             ),
        #             scalar_run_args=", ".join(
        #                 [
        #                     "ctypes.{ctype_name}({par_name})".format(
        #                         ctype_name=np.ctypeslib.as_ctypes_type(
        #                             self.implementation_ir.parameters[n].data_type.dtype
        #                         ).__name__,
        #                         par_name=n,
        #                     )
        #                     for n in sorted(self.implementation_ir.parameters)
        #                 ]
        if self.implementation_ir.multi_stages:
            source = (
                (
                    "\nI=np.int32(_domain_[0])\nJ=np.int32(_domain_[1])\nK=np.int32(_domain_[2])\n"
                    + total_field_sizes
                    + "\n"
                    + "\n".join(field_slices)
                    + """
#dace_lib['__dace_init_{program_name}']({run_args}, {total_field_sizes})
#if exec_info is not None:
#    exec_info['pyext_program_start_time'] = time.perf_counter()
#dace_lib['__program_{program_name}']({run_args}, {total_field_sizes})
#if exec_info is not None:
#    exec_info['pyext_program_end_time'] = time.perf_counter()
#dace_lib['__dace_exit_{program_name}']({run_args}, {total_field_sizes})
dace_lib['__dace_init_{program_name}']({run_args})
if exec_info is not None:
    exec_info['pyext_program_start_time'] = time.perf_counter()
dace_lib['__program_{program_name}']({run_args})
if exec_info is not None:
    exec_info['pyext_program_end_time'] = time.perf_counter()
dace_lib['__dace_exit_{program_name}']({run_args})
""".format(
                        program_name=self.implementation_ir.sdfg.name,
                        run_args=", ".join(run_args_strs),
                        total_field_sizes=",".join(
                            v for k, v in sorted(symbol_ctype_strs.items())
                        ),
                    )
                )
                + "\n".join(
                    [
                        "#{name}[{slice}] = {name}_copy".format(
                            name=name,
                            slice=",".join(
                                f'_origin_["{name}"][{i}] - {-self.implementation_ir.fields_extents[name][i][0]}:_origin_["{name}"][{i}] - {-self.implementation_ir.fields_extents[name][i][0]}+_domain_[{i}] + {self.implementation_ir.fields_extents[name].frame_size[i]}'
                                for i in range(len(self.implementation_ir.fields[name].axes))
                            ),
                        )
                        for name in self.implementation_ir.arg_fields
                    ]
                )
            )

        else:
            source = "\n"

        source = source + (
            """
if exec_info is not None:
    exec_info["run_end_time"] = time.perf_counter()
"""
        )
        sources.extend(source.splitlines())

        return sources.text


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

    GENERATOR_CLASS = DacePyModuleGenerator

    @classmethod
    def get_dace_module_path(cls, stencil_id):
        path = os.path.join(
            cls.get_stencil_package_path(stencil_id),
            cls.get_stencil_module_name(stencil_id, qualified=False),
        )

        return path

    @classmethod
    def generate_dace(cls, stencil_id, implementation_ir, options):
        sdfg = SDFGBuilder.apply(implementation_ir)

        dace_build_path = os.path.relpath(cls.get_dace_module_path(stencil_id))
        os.makedirs(dace_build_path, exist_ok=True)

        # sdfg = performance_ir.sdfg

        import json

        from dace.transformation.dataflow.merge_arrays import MergeArrays

        save_sdfgs = {}

        sdfg.save(dace_build_path + os.path.sep + "00_raw.sdfg")

        sdfg.apply_transformations_repeated(MergeArrays)
        sdfg.validate()
        from dace.transformation.interstate import StateFusion

        sdfg.apply_transformations_repeated([StateFusion], strict=True, validate=False)

        sdfg.save(dace_build_path + os.path.sep + "01_fused_states.sdfg")
        cls.transform_library(sdfg)
        sdfg.save(dace_build_path + os.path.sep + "02_library_nodes_optimized.sdfg")
        sdfg.expand_library_nodes()
        from dace.transformation.interstate import InlineSDFG

        sdfg.apply_transformations_repeated([InlineSDFG], validate=False)
        sdfg.save(dace_build_path + os.path.sep + "03_library_expanded.sdfg")
        cls.transform_to_device(sdfg)
        # sdfg.save(dace_build_path + os.path.sep + "04_on_device.sdfg")
        cls.transform_optimize(sdfg)
        # sdfg.save(dace_build_path + os.path.sep + "05_optimized.sdfg")
        sdfg.validate()
        implementation_ir.sdfg = sdfg

        program_folder = dace.codegen.compiler.generate_program_folder(
            sdfg=sdfg, code_objects=generate_code(sdfg), out_path=dace_build_path
        )
        assert program_folder == dace_build_path
        dace_ext_lib = dace.codegen.compiler.configure_and_compile(program_folder)

        return dace_ext_lib, dace_build_path

    @classmethod
    def generate(cls, stencil_id, definition_ir, definition_func, options):
        from gt4py import gt_src_manager

        cls._check_options(options)
        implementation_ir = gt_analysis.transform(definition_ir, options)

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

        return super(DaceBackend, cls)._generate_module(
            stencil_id, implementation_ir, definition_func, generator_options, extra_cache_info
        )

    @classmethod
    def transform_library(cls, sdfg):
        raise NotImplementedError("This method must be implemented in a subclass.")

    @classmethod
    def transform_to_device(cls, sdfg):
        raise NotImplementedError("This method must be implemented in a subclass.")

    @classmethod
    def transform_optimize(cls, sdfg):
        raise NotImplementedError("This method must be implemented in a subclass.")
