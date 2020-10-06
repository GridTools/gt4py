import os
import copy

import numpy as np

import dace
from dace.codegen.codegen import generate_code
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL
import dace.codegen.instrumentation

import gt4py.analysis as gt_analysis
from gt4py import backend as gt_backend
from gt4py.ir import DataType, IterationOrder
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
        dace_ext_lib = self.options.dace_ext_lib.replace("\\", "\\\\")
        source = f"""
import functools
import ctypes
import os
import weakref

dace_lib = ctypes.CDLL("{dace_ext_lib}")
"""
        return source

    def generate_class_members(self):
        sources = gt_text.TextBlock(
            indent_size=gt_backend.BaseModuleGenerator.TEMPLATE_INDENT_SIZE
        )
        dummy_args = ", ".join("None" for _ in self.implementation_ir.sdfg.arglist())
        source = f"""
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    dace_lib["__dace_init_{self.implementation_ir.sdfg.name}"]({dummy_args})
    weakref.finalize(self, dace_lib["__dace_exit_{self.implementation_ir.sdfg.name}"], {dummy_args})
"""
        sources.extend(source.splitlines())
        return sources.text

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
                    """{name}_interface = {name}.{interface}
itemsize = np.dtype({name}_interface['descr'][0][1]).itemsize
if {name}_interface['strides'] is None:
    _{name}_K_stride = 1
    _{name}_J_stride = {name}_interface['shape'][2]
    _{name}_I_stride = {name}_interface['shape'][1] * _{name}_J_stride
else:
    _{name}_I_stride, _{name}_J_stride, _{name}_K_stride = [np.int32(st//itemsize) for st in {name}_interface['strides']]
assert not {name}_interface['data'][1] # assert not readonly
{name}_ptr = int({name}_interface['data'][0] + sum([(o-e)*st for o, e, st in zip(_origin_['{name}'], {extent}, [_{name}_I_stride*itemsize, _{name}_J_stride*itemsize, _{name}_K_stride*itemsize])]))""".format(
                        name=field_name,
                        interface=self.array_interface_name,
                        extent=tuple(
                            -self.implementation_ir.fields_extents[field_name][i][0]
                            for i in range(3)
                        ),
                    )
                )
        # total_field_sizes = []
        symbol_ctype_strs = dict(I="ctypes.c_int(I)", J="ctypes.c_int(J)", K="ctypes.c_int(K)")
        for field_name in self.implementation_ir.arg_fields:
            if field_name not in self.implementation_ir.unreferenced:
                # total_field_sizes.append(
                #     f"_{field_name}_I_stride, _{field_name}_J_stride, _{field_name}_K_stride = tuple(np.int32(s/{field_name}.itemsize) for s in {field_name}.strides)"
                # )
                symbol_ctype_strs[
                    f"_{field_name}_I_stride"
                ] = f"ctypes.c_int(_{field_name}_I_stdride)"
                symbol_ctype_strs[
                    f"_{field_name}_J_stride"
                ] = f"ctypes.c_int(_{field_name}_J_stride)"
                symbol_ctype_strs[
                    f"_{field_name}_K_stride"
                ] = f"ctypes.c_int(_{field_name}_K_stride)"

        # total_field_sizes = "\n".join(total_field_sizes)
        run_args_names = sorted(args)
        run_args_strs = []
        for name, datadescr in self.implementation_ir.sdfg.arglist().items():
            if name in self.implementation_ir.arg_fields:
                run_args_strs.append(f"ctypes.c_void_p({name}_ptr)")
            else:
                run_args_strs.append(
                    "ctypes.{ctype_name}(np.{numpy_type}({par_name}))".format(
                        ctype_name=DataType.from_dtype(datadescr.dtype.type).ctypes_str,
                        numpy_type=datadescr.dtype.type.__name__,
                        par_name=name if not name.startswith("_gt_loc__") else name[9:],
                    )
                )

        if self.implementation_ir.multi_stages:
            source = (
                "\nI=np.int32(_domain_[0])\nJ=np.int32(_domain_[1])\nK=np.int32(_domain_[2])\n"
                # + total_field_sizes
                # + "\n"
                + "\n".join(field_slices)
                + """
if exec_info is not None:
    exec_info['pyext_program_start_time'] = time.perf_counter()
dace_lib['__program_{program_name}']({run_args})
if exec_info is not None:
    exec_info['pyext_program_end_time'] = time.perf_counter()
    path = os.path.join('{build_path}', 'perf')
    files = [f for f in os.listdir(path) if f.startswith('report-')]
    exec_info['instrumentation_report'] = os.path.join(path,sorted(files, reverse=True)[0])
""".format(
                    program_name=self.implementation_ir.sdfg.name,
                    run_args=", ".join(run_args_strs),
                    total_field_sizes=",".join(v for k, v in sorted(symbol_ctype_strs.items())),
                    build_path=os.path.abspath(self.options.dace_build_path).replace("\\", "\\\\"),
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


class DaceOptimizer:

    description = "no optimization"

    def transform_library(self, sdfg):
        return sdfg

    def transform_to_device(self, sdfg):
        return sdfg

    def transform_optimize(self, sdfg):
        return sdfg


class CPUDaceOptimizer(DaceOptimizer):
    description = "no optimization on CPU"

    def transform_to_device(self, sdfg):
        for state in sdfg.nodes():
            state.instrument = dace.InstrumentationType.Timer
        return sdfg


class CudaDaceOptimizer(DaceOptimizer):
    description = "no optimization on GPU"

    def transform_to_device(self, sdfg):
        import dace
        from dace.transformation.dataflow import MapCollapse

        sdfg.apply_transformations_repeated(MapCollapse, validate=False)

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
                    self.transform_to_device(node.sdfg)

        from dace.sdfg.utils import consolidate_edges

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    consolidate_edges(node.sdfg)
        for state in sdfg.nodes():
            state.instrument = dace.InstrumentationType.GPU_Events
        return sdfg


class SDFGInjector(DaceOptimizer):

    description = "externally provided SDFG"

    def __init__(self, sdfg):
        if isinstance(sdfg, str):
            sdfg = dace.SDFG.from_file(sdfg)
        self.sdfg: dace.SDFG = sdfg

    def transform_optimize(self, sdfg):

        if len(sdfg.signature_arglist()) != len(self.sdfg.signature_arglist()):
            raise ValueError(
                "SDFG to inject does not have matching signature length. ({here} != {raw})".format(
                    here=len(self.sdfg.signature_arglist()), raw=len(sdfg.signature_arglist())
                )
            )
        if not all(
            here == res
            for here, res in zip(sdfg.signature_arglist(), self.sdfg.signature_arglist())
        ):
            l = list(
                (here, res)
                for here, res in zip(sdfg.signature_arglist(), self.sdfg.signature_arglist())
                if here != res
            )
            raise ValueError(
                "SDFG to inject does not have matching signature. ({here} != {raw})".format(
                    here=l[0][0], raw=l[0][1]
                )
            )
        res = copy.deepcopy(self.sdfg)
        return res


class DaceBackend(gt_backend.BaseBackend):
    name = "dace"
    options = {
        "optimizer": {"versioning": True},
        "save_intermediate": {"versioning": True},
        "validate": {"versioning": True},
        "enforce_dtype": {"versioning": True},
        "specialize_sdfg_vars": {"versioning": True},
        "gpu_block_size": {"versioning": True},
        "computation_layout": {"versioning": True},
    }
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": dace_layout,
        "is_compatible_layout": dace_is_compatible_layout,
        "is_compatible_type": dace_is_compatible_type,
    }

    GENERATOR_CLASS = DacePyModuleGenerator
    DEFAULT_OPTIMIZER = CPUDaceOptimizer()

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
        from gt4py.backend.dace.sdfg.library.nodes import StencilLibraryNode

        comp_layout = options.backend_opts.get("computation_layout", "JKI")
        for name, array in sdfg.arrays.items():
            import dace.data

            if isinstance(array, dace.data.Array) and array.transient:
                layout = cls.storage_info["layout_map"]([True] * 3)
                stride = 1
                strides = [None] * 3
                for i in reversed(np.argsort(layout)):
                    strides[i] = stride
                    stride = stride * array.shape[i]
                array.strides = strides

                for state in sdfg.nodes():
                    for node in state.nodes():
                        if isinstance(node, StencilLibraryNode):
                            node.loop_order = comp_layout

                            for interval in node.intervals:
                                if name in interval.sdfg.arrays:
                                    interval.sdfg.arrays[name].strides = strides

        dace_build_path = os.path.relpath(cls.get_dace_module_path(stencil_id))
        os.makedirs(dace_build_path, exist_ok=True)

        save = options.backend_opts.get("save_intermediate", False)
        optimizer = options.backend_opts.get("optimizer", cls.DEFAULT_OPTIMIZER)
        if "optimizer" in options.backend_opts:
            options.backend_opts["optimizer"] = optimizer.__class__.__name__
        validate = options.backend_opts.get("validate", True)

        from dace.transformation.dataflow.merge_arrays import InMergeArrays

        if save:
            sdfg.save(dace_build_path + os.path.sep + "00_raw.sdfg")

        sdfg.apply_transformations_repeated(InMergeArrays)
        from dace.transformation.interstate import StateFusion

        sdfg.apply_transformations_repeated([StateFusion], strict=False, validate=False)

        if save:
            sdfg.save(dace_build_path + os.path.sep + "01_fused_states.sdfg")

        sdfg = optimizer.transform_library(sdfg)

        if save:
            sdfg.save(dace_build_path + os.path.sep + "02_library_nodes_optimized.sdfg")

        sdfg.expand_library_nodes()
        from dace.transformation.interstate import InlineSDFG

        sdfg.apply_transformations_repeated([InlineSDFG], validate=False)

        if save:
            sdfg.save(dace_build_path + os.path.sep + "03_library_expanded.sdfg")

        sdfg = optimizer.transform_to_device(sdfg)

        if save:
            sdfg.save(dace_build_path + os.path.sep + "04_on_device.sdfg")

        sdfg = optimizer.transform_optimize(sdfg)

        if save:
            sdfg.save(dace_build_path + os.path.sep + "05_optimized.sdfg")

        if validate:
            sdfg.validate()

        specialize_dict = options.backend_opts.get("specialize_sdfg_vars", {})
        block_size_str = options.backend_opts.get("gpu_block_size", "64,2,1")
        dace.Config.set("compiler", "cuda", "default_block_size", value=block_size_str)
        sdfg: dace.SDFG
        sdfg.specialize(specialize_dict)
        for sd in sdfg.all_sdfgs_recursive():
            sd.specialize(specialize_dict)
        sdfg.save(dace_build_path + os.path.sep + "tmp.sdfg")
        sdfg = dace.SDFG.from_file(dace_build_path + os.path.sep + "tmp.sdfg")

        implementation_ir.sdfg = copy.deepcopy(sdfg)

        sdfg.build_folder = os.path.abspath(dace_build_path)
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
