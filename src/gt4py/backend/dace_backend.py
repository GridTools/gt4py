import os

import itertools

import numpy as np

from gt4py import backend as gt_backend
from gt4py import analysis as gt_analysis
from gt4py.utils import text as gt_text
import gt4py.ir as gt_ir
import dace
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL
from dace.sdfg import SDFG
from gt4py.utils.attrib import (
    attribute,
    attribkwclass as attribclass,
    attributes_of,
    Any as Any,
    Dict as DictOf,
    List as ListOf,
    Tuple as TupleOf,
    Union as UnionOf,
    Optional as OptionalOf,
)

LOADED_REGISTRY = dict()


def local_name(name, offset, is_target):
    CONFIG = dict(local_name_prefix="_gt_loc_", local_name_prefix_out="_gt_loc_out_")
    prefix = CONFIG["local_name_prefix_out"] if is_target else CONFIG["local_name_prefix"]

    offset_strs = []
    offset = offset or dict()

    for k, v in offset.items():
        offset_strs.append(k + ("p" if v >= 0 else "n") + str(abs(v)))
    offset_str = "_" + "".join(offset_strs) if offset_strs else ""
    local_name = f"{prefix}{name}{offset_str}"

    return local_name


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


@attribclass
class MappedMemletInfo:
    outer_name = attribute(of=str)
    local_name = attribute(of=str)
    subset_str = attribute(of=str, optional=True)
    offset = attribute(of=dict)
    num = attribute(of=int)


class SDFGBuilder:
    class GenerateIterationRangePass(gt_ir.IRNodeVisitor):
        @classmethod
        def apply(cls, iir):
            instance = cls(iir)
            instance.visit(iir)

        def __init__(self, iir):
            self.iir = iir
            self.visitor_state = dict(i_range=None, j_range=None)

        def visit_Stage(self, node: gt_ir.Stage):
            i_extent = node.compute_extent[0]
            j_extent = node.compute_extent[1]
            self.visitor_state["i_range"] = i_extent
            self.visitor_state["j_range"] = j_extent
            # self.visitor_state["i_range"] = "{lb}:I{rb}".format(
            #     lb=i_extent[0], rb=(" + " + str(i_extent[1])) if i_extent[1] else "",
            # )
            # self.visitor_state["j_range"] = "{lb}:J{rb}".format(
            #     lb=str(j_extent[0]), rb=(" + " + str(j_extent[1])) if j_extent[1] else "",
            # )
            self.generic_visit(node)
            self.visitor_state["i_range"] = None
            self.visitor_state["j_range"] = None

        def visit_ApplyBlock(self, node: gt_ir.ApplyBlock):
            node.i_range = self.visitor_state["i_range"]
            node.j_range = self.visitor_state["j_range"]
            k_start = (
                node.interval.start.offset
                if node.interval.start.level == gt_ir.LevelMarker.START
                else "K{:+d}".format(node.interval.start.offset)
            )
            k_end = (
                node.interval.end.offset
                if node.interval.end.level == gt_ir.LevelMarker.START
                else "K{:+d}".format(node.interval.end.offset)
            )
            node.k_range = (k_start, k_end)

            # node.k_range_tuple = (str(k_extent[0]), 'K+'+str(k_extent[1]))
            #     = "K{l}:K{r}".format(
            #     l=(" - " + str(-j_extent[0]) if j_extent[0] else ""),
            #     r=(" + " + str(j_extent[1]) if j_extent[1] else ""),
            # )
            # self.generic_visit(node)

    class GenerateTaskletInfoPass(gt_ir.IRNodeVisitor):
        @classmethod
        def apply(cls, iir: gt_ir.StencilImplementation):
            transormer = cls()
            transormer.visit(iir)

        def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
            self.parameters = node.parameters
            self.generic_visit(node)

        def visit_ApplyBlock(self, node: gt_ir.ApplyBlock):
            self.tasklet_targets = set()
            self.visit(node.body)
            self.tasklet_targets = None

        def visit_Assign(self, node: gt_ir.Assign):
            self.visit(node.value)
            self.visit(node.target, is_target=True)

        def visit_FieldRef(self, node: gt_ir.FieldRef, is_target=False):
            key = (node.name, tuple((k, v) for k, v in node.offset.items()))
            if is_target:
                self.tasklet_targets.add(key)
            node.was_output = key in self.tasklet_targets
            node.local_name = local_name(node.name, node.offset, node.was_output)

        def visit_VarRef(self, node: gt_ir.VarRef, is_target=False):
            if is_target:
                self.tasklet_targets.add((node.name, None))
            node.was_output = (node.name, None) in self.tasklet_targets
            node.local_name = local_name(node.name, None, node.was_output)

    class GenerateMappedMemletsPass(gt_ir.IRNodeVisitor):
        @classmethod
        def apply(cls, iir: gt_ir.StencilImplementation):
            transformer = cls()
            transformer.input_memlets = None
            transformer.output_memlets = None
            transformer.visit(iir)

        def visit_Assign(self, node: gt_ir.Assign):
            assert isinstance(node.target, (gt_ir.FieldRef, gt_ir.VarRef))
            self.visit(node.value)
            self.visit(node.target)

        def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
            self.fields = node.fields
            self.field_extents = node.fields_extents
            self.parameters = node.parameters
            self.generic_visit(node)

        def visit_ApplyBlock(self, node: gt_ir.ApplyBlock):
            self.input_memlets = dict()
            self.output_memlets = dict()

            self.visit(node.body)

            node.mapped_input_memlets = dict()
            node.mapped_output_memlets = dict()
            node.mapped_input_memlet_infos = self.input_memlets
            node.mapped_output_memlet_infos = self.output_memlets

            for memlet_name, memlet_info in self.input_memlets.items():
                node.mapped_input_memlets[memlet_name] = dace.Memlet.simple(
                    memlet_info.outer_name, memlet_info.subset_str, num_accesses=memlet_info.num
                )
            for memlet_name, memlet_info in self.output_memlets.items():

                node.mapped_output_memlets[memlet_name] = dace.Memlet.simple(
                    memlet_info.outer_name, memlet_info.subset_str, num_accesses=memlet_info.num
                )

            self.input_memlets = None
            self.output_memlets = None
            self.tasklet_targets = None

        def visit_FieldRef(self, node: gt_ir.FieldRef):
            key = node.local_name
            memlet_dict = self.output_memlets if node.was_output else self.input_memlets
            if key not in memlet_dict:
                subset_list = []
                itervar = dict(I="i", J="j", K="k")
                itervar_idx = dict(I=0, J=1, K=2)
                origin = self.field_extents[node.name].lower_indices
                for k, v in node.offset.items():
                    if k != "K":
                        subset_list.append(itervar[k] + "{:+d}".format(v - origin[itervar_idx[k]]))
                    else:
                        subset_list.append(itervar[k] + "{:+d}".format(v))
                subset_str = ", ".join(subset_list) if subset_list else "0"

                memlet_dict[key] = MappedMemletInfo(
                    num=1,
                    outer_name=node.name,
                    local_name=key,
                    subset_str=subset_str,
                    offset=node.offset,
                )

        def visit_VarRef(self, node: gt_ir.VarRef):
            if node.name in self.parameters:
                key = node.local_name
                memlet_dict = self.output_memlets if node.was_output else self.input_memlets
                if key not in memlet_dict:
                    memlet_dict[key] = MappedMemletInfo(
                        num=1, outer_name=node.name, local_name=key, subset_str="0", offset=dict()
                    )

    class GenerateTaskletSourcePass(gt_ir.IRNodeVisitor):

        OP_TO_PYTHON = {
            **gt_ir.UnaryOperator.IR_OP_TO_PYTHON_SYMBOL,
            **gt_ir.BinaryOperator.IR_OP_TO_PYTHON_SYMBOL,
        }

        @classmethod
        def apply(cls, iir: gt_ir.StencilImplementation):
            transformer = cls()
            transformer.parameters = iir.parameters
            transformer.visit(iir)

        def visit_ApplyBlock(self, node: gt_ir.ApplyBlock):
            self.tasklet_targets = set()
            sources = self.visit(node.body)
            node.tasklet_code = sources.text
            self.tasklet_targets = None

        ###from debug backend
        def visit_Assign(self, node: gt_ir.Assign):
            assert isinstance(node.target, (gt_ir.FieldRef, gt_ir.VarRef))
            lhs = self.visit(node.target)
            rhs = self.visit(node.value)
            source = f"{lhs} = {rhs}"

            return source

        def visit_BlockStmt(self, node: gt_ir.BlockStmt):
            body_sources = gt_text.TextBlock()
            for stmt in node.stmts:
                stmt_source = self.visit(stmt)
                if isinstance(stmt_source, list):
                    body_sources.extend(stmt_source)
                else:
                    body_sources.append(stmt_source)

            return body_sources

        def visit_ScalarLiteral(self, node: gt_ir.ScalarLiteral):
            return str(node.value)

        def visit_VarRef(self, node: gt_ir.VarRef):
            return node.local_name

        def visit_FieldRef(self, node: gt_ir.FieldRef):
            return node.local_name

        def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr):
            fmt = "({})" if isinstance(node.arg, gt_ir.CompositeExpr) else "{}"
            source = "{op}{expr}".format(
                op=self.OP_TO_PYTHON[node.op], expr=fmt.format(self.visit(node.arg))
            )
            return source

        def visit_BinOpExpr(self, node: gt_ir.BinOpExpr):
            lhs_fmt = "({})" if isinstance(node.lhs, gt_ir.CompositeExpr) else "{}"
            rhs_fmt = "({})" if isinstance(node.rhs, gt_ir.CompositeExpr) else "{}"
            source = "{lhs} {op} {rhs}".format(
                lhs=lhs_fmt.format(self.visit(node.lhs)),
                op=self.OP_TO_PYTHON[node.op],
                rhs=rhs_fmt.format(self.visit(node.rhs)),
            )

            return source

        def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr):
            then_fmt = "({})" if isinstance(node.then_expr, gt_ir.CompositeExpr) else "{}"
            else_fmt = "({})" if isinstance(node.else_expr, gt_ir.CompositeExpr) else "{}"
            source = "{then_expr} if {condition} else {else_expr}".format(
                condition=self.visit(node.condition),
                then_expr=then_fmt.format(self.visit(node.then_expr)),
                else_expr=else_fmt.format(self.visit(node.else_expr)),
            )

            return source

        def visit_If(self, node: gt_ir.If):
            body_sources = gt_text.TextBlock()
            body_sources.append("if {condition}:".format(condition=self.visit(node.condition)))
            body_sources.indent()
            for stmt in node.main_body.stmts:
                body_sources.extend(self.visit(stmt))
            body_sources.dedent()
            if node.else_body:
                body_sources.append("else:")
                body_sources.indent()

                for stmt in node.else_body.stmts:
                    body_sources.extend(self.visit(stmt))
                body_sources.dedent()
            return ["".join([str(item) for item in line]) for line in body_sources.lines]

    class GenerateSDFGPass(gt_ir.IRNodeVisitor):
        @classmethod
        def apply(cls, iir: gt_ir.StencilImplementation):
            transformer = cls(iir.name.split(".")[-1])
            transformer.visit(iir)
            return transformer.sdfg

        def __init__(self, name):
            self.sdfg = dace.SDFG(name)
            self.iteration_order = None

            self.tail_state = self.sdfg.add_state(is_start_state=True)
            I = dace.symbol("I")
            J = dace.symbol("J")
            K = dace.symbol("K")
            self.map_counter = itertools.count()
            self.tasklet_counter = itertools.count()
            self.symbolic_domain = (I, J, K)

        def new_tasklet_name(self):
            return "tasklet_" + str(next(self.tasklet_counter))

        def new_map_name(self):
            return "map_" + str(next(self.map_counter))

        def _append_states(self, entry_state, exit_state):
            self.sdfg.add_edge(self.tail_state, entry_state, dace.InterstateEdge())
            self.tail_state = exit_state

        #
        def _make_parallel_computation(self, node: gt_ir.ApplyBlock):
            state = self.sdfg.add_state()

            tasklet = state.add_tasklet(
                name=self.new_tasklet_name(),
                inputs=node.mapped_input_memlets,
                outputs=node.mapped_output_memlets,
                code=node.tasklet_code,
            )

            range = dict(
                i="{:d}:I{:+d}".format(*node.i_range,),
                j="{:d}:J{:+d}".format(*node.j_range),
                k="{}:{}".format(*node.k_range),
            )
            map_entry, map_exit = state.add_map(name=self.new_map_name(), ndrange=range)

            # in_field_accessors = dict()
            for memlet_info in node.mapped_input_memlet_infos.values():
                name = memlet_info.outer_name
                # if name not in in_field_accessors:
                #     in_field_accessors[name] = state.add_read(name)
                state.add_memlet_path(
                    state.add_read(name),
                    map_entry,
                    tasklet,
                    memlet=node.mapped_input_memlets[memlet_info.local_name],
                    dst_conn=memlet_info.local_name,
                )
            if len(node.mapped_input_memlet_infos) == 0:
                state.add_edge(map_entry, None, tasklet, None, dace.EmptyMemlet())

            out_field_accessors = dict()
            for memlet_info in node.mapped_output_memlet_infos.values():
                name = memlet_info.outer_name
                if name not in out_field_accessors:
                    out_field_accessors[name] = state.add_write(name)
                state.add_memlet_path(
                    tasklet,
                    map_exit,
                    out_field_accessors[name],
                    memlet=node.mapped_output_memlets[memlet_info.local_name],
                    src_conn=memlet_info.local_name,
                )
            if len(node.mapped_output_memlet_infos) == 0:
                state.add_edge(tasklet, None, map_exit, None, dace.EmptyMemlet())

            return state, state

        def _make_ij_computation(self, node: gt_ir.ApplyBlock):
            state = self.sdfg.add_state()

            tasklet = state.add_tasklet(
                name=self.new_tasklet_name(),
                inputs=node.mapped_input_memlets,
                outputs=node.mapped_output_memlets,
                code=node.tasklet_code,
            )

            range = dict(
                i="{:d}:I{:+d}".format(*node.i_range,), j="{:d}:J{:+d}".format(*node.j_range)
            )
            map_entry, map_exit = state.add_map(name=self.new_map_name(), ndrange=range)

            in_field_accessors = dict()
            for memlet_info in node.mapped_input_memlet_infos.values():
                name = memlet_info.outer_name
                if name not in in_field_accessors:
                    in_field_accessors[name] = state.add_read(name)
                state.add_memlet_path(
                    in_field_accessors[name],
                    map_entry,
                    tasklet,
                    memlet=node.mapped_input_memlets[memlet_info.local_name],
                    dst_conn=memlet_info.local_name,
                )
            out_field_accessors = dict()
            for memlet_info in node.mapped_output_memlet_infos.values():
                name = memlet_info.outer_name
                if name not in out_field_accessors:
                    out_field_accessors[name] = state.add_write(name)
                state.add_memlet_path(
                    tasklet,
                    map_exit,
                    out_field_accessors[name],
                    memlet=node.mapped_output_memlets[memlet_info.local_name],
                    src_conn=memlet_info.local_name,
                )
            return state

        def _make_forward_computation(self, node: gt_ir.ApplyBlock):
            state = self._make_ij_computation(node)
            loop_start, _, loop_end = self.sdfg.add_loop(
                None, state, None, "k", str(node.k_range[0]), f"k<{node.k_range[1]}", "k+1"
            )

            return loop_start, loop_end

        def _make_backward_computation(self, node: gt_ir.ApplyBlock):
            state = self._make_ij_computation(node)
            loop_start, _, loop_end = self.sdfg.add_loop(
                None, state, None, "k", str(node.k_range[1]) + "-1", f"k>={node.k_range[0]}", "k-1"
            )

            return loop_start, loop_end

        def visit_ApplyBlock(self, node: gt_ir.ApplyBlock):

            if self.iteration_order == gt_ir.IterationOrder.PARALLEL:
                make_computation = self._make_parallel_computation
            if self.iteration_order == gt_ir.IterationOrder.FORWARD:
                make_computation = self._make_forward_computation
            if self.iteration_order == gt_ir.IterationOrder.BACKWARD:
                make_computation = self._make_backward_computation

            self._append_states(*make_computation(node))

        def visit_MultiStage(self, node: gt_ir.MultiStage):
            self.iteration_order = node.iteration_order
            self.generic_visit(node)
            self.iteration_order = None

        def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):

            for field in node.fields.values():
                if field.name in node.arg_fields:
                    I = dace.symbol(f"_{field.name}_I")
                    J = dace.symbol(f"_{field.name}_J")
                    K = dace.symbol(f"_{field.name}_K")
                    add_to_sdfg = lambda *args, **kwargs: self.sdfg.add_array(
                        *args, strides=(J * K, K, 1), **kwargs
                    )
                else:
                    assert field.name in node.temporary_fields
                    add_to_sdfg = lambda *args, **kwargs: self.sdfg.add_transient(
                        *args, toplevel=True, **kwargs
                    )
                shape = [
                    d + f
                    for d, f in zip(
                        self.symbolic_domain, node.fields_extents[field.name].frame_size
                    )
                ]
                add_to_sdfg(field.name, shape=shape, dtype=field.data_type.dtype.type)
            for k, v in node.parameters.items():
                self.sdfg.add_scalar(k, v.data_type.dtype.type)
            self.generic_visit(node)

    @classmethod
    def apply(cls, iir):
        cls.GenerateIterationRangePass.apply(iir)
        cls.GenerateTaskletInfoPass.apply(iir)
        cls.GenerateMappedMemletsPass.apply(iir)
        cls.GenerateTaskletSourcePass.apply(iir)
        sdfg = cls.GenerateSDFGPass.apply(iir)
        return sdfg


# import ctypes
# def load_dace_program(dace_ext_lib, stencil_name):
#     dll = ctypes.CDLL(dace_ext_lib)
#     return getattr(dll, f"__program_{stencil_name}")

# arrays = (,)
# symbols = (,)
# args = (*sorted(arrays), *sorted(symbols))
# dll.__dace_init(*args)
# dll.__program_NAMEOFPROGRAM(*args)
# dll.__dace_exit(*args)


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
        field_slices = []

        for field_name in self.performance_ir.arg_fields:
            if field_name not in self.performance_ir.unreferenced:
                field_slices.append(
                    """#{name}_copy = np.array({name}[{slice}], copy=True)
#assert {name}_copy.shape=={shape}, str({name}_copy.shape)+'!='+str({shape})
#assert {name}_copy.strides==tuple({name}_copy.itemsize*s for s in {strides}), str({name}_copy.strides)+'!='+str(tuple({name}.itemsize*s for s in {strides}))
{name} = {name}[{slice}] if {name} is not None else None
if {name} is not None:
    assert {name}.shape=={shape}, str({name}.shape)+'!='+str({shape})
    assert {name}.strides==tuple({name}.itemsize*s for s in {strides}), str({name}.strides)+'!='+str(tuple({name}.itemsize*s for s in {strides}))""".format(
                        name=field_name,
                        slice=",".join(
                            f'_origin_["{field_name}"][{i}] - {-self.performance_ir.fields_extents[field_name][i][0]}:_origin_["{field_name}"][{i}] - {-self.performance_ir.fields_extents[field_name][i][0]}+_domain_[{i}] + {self.performance_ir.fields_extents[field_name].frame_size[i]}'
                            if self.performance_ir.fields[field_name].axes[i]
                            != self.performance_ir.domain.sequential_axis.name
                            else f'_origin_["{field_name}"][{i}]:_origin_["{field_name}"][{i}]+_domain_[{i}]'
                            for i in range(len(self.performance_ir.fields[field_name].axes))
                        ),
                        shape=tuple(
                            s
                            if self.performance_ir.fields[field_name].axes[i]
                            != self.performance_ir.domain.sequential_axis.name
                            else dace.symbol("K")
                            for i, s in enumerate(
                                self.performance_ir.sdfg.arrays[field_name].shape
                            )
                        ),
                        strides=self.performance_ir.sdfg.arrays[field_name].strides,
                    )
                )
        total_field_sizes = []
        for field_name in self.performance_ir.arg_fields:
            if field_name not in self.performance_ir.unreferenced:
                total_field_sizes.append(
                    f"_{field_name}_K = np.int32({field_name}.strides[1])/{field_name}.itemsize if not {field_name} is None else 0.0"
                )
                total_field_sizes.append(
                    f"_{field_name}_J = np.int32({field_name}.strides[0]/_{field_name}_K)/{field_name}.itemsize if not {field_name} is None else 0.0"
                )
                total_field_sizes.append(
                    f"_{field_name}_I = np.int32({field_name}.strides[0]/_{field_name}_K)/{field_name}.itemsize if not {field_name} is None else 0.0"
                )
        total_field_sizes = "\n".join(total_field_sizes)
        source = (
            (
                "\nI=np.int32(_domain_[0])\nJ=np.int32(_domain_[1])\nK=np.int32(_domain_[2])\n"
                + total_field_sizes
                + "\n"
                + "\n".join(field_slices)
                + """
dace_program({run_args}, I=I, J=J, K=K, {total_field_sizes})
""".format(
                    run_args=", ".join(
                        [
                            f"{n}={n}" if n in self.performance_ir.arg_fields else f"{n}={n}"
                            for n in args
                        ]
                    ),
                    total_field_sizes=", ".join(
                        f"_{field_name}_I=np.int32(_{field_name}_I), _{field_name}_J=np.int32(_{field_name}_J), _{field_name}_K=np.int32(_{field_name}_K)"
                        for field_name in self.performance_ir.arg_fields
                        if field_name not in self.performance_ir.unreferenced
                    ),
                )
            )
            + "\n".join(
                [
                    "#{name}[{slice}] = {name}_copy".format(
                        name=name,
                        slice=",".join(
                            f'_origin_["{name}"][{i}] - {-self.performance_ir.fields_extents[name][i][0]}:_origin_["{name}"][{i}] - {-self.performance_ir.fields_extents[name][i][0]}+_domain_[{i}] + {self.performance_ir.fields_extents[name].frame_size[i]}'
                            for i in range(len(self.performance_ir.fields[name].axes))
                        ),
                    )
                    for name in self.performance_ir.arg_fields
                ]
            )
        )

        source = source + (
            """
if exec_info is not None:
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
        sdfg = SDFGBuilder.apply(implementation_ir)
        # sdfg = performance_ir.sdfg

        import json

        with open("tmp.sdfg", "w") as sdfgfile:
            json.dump(sdfg.to_json(), sdfgfile)
        sdfg.apply_strict_transformations()
        with open("tmp.sdfg", "w") as sdfgfile:
            json.dump(sdfg.to_json(), sdfgfile)
        import dace.graph.labeling

        dace.graph.labeling.propagate_labels_sdfg(sdfg)
        sdfg.validate()

        performance_ir.sdfg = sdfg
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
