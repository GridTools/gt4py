import os
import ctypes

import itertools
from collections import OrderedDict

import numpy as np

from gt4py import backend as gt_backend
from gt4py import analysis as gt_analysis
from gt4py.utils import text as gt_text
import gt4py.ir as gt_ir
import dace
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL
from dace.codegen.codegen import generate_code
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

from dace import registry, dtypes
from dace.codegen.targets.cpu import CPUCodeGen
from dace.codegen.targets.target import TargetCodeGenerator, DefinedType
from dace.codegen.targets.common import sym2cpp

TargetCodeGenerator.unregister(CPUCodeGen)

from dace.sdfg import SDFG


@registry.autoregister_params(name="cpu")
class CPUWithPersistent(CPUCodeGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocated_symbols = set()

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):

        name = node.data
        nodedesc = node.desc(sdfg)

        if nodedesc.transient is False:
            return

        if nodedesc.storage == dtypes.StorageType.CPU_Heap:
            # Check if array is already allocated
            try:
                self._dispatcher.defined_vars.get(name)
                return  # Array was already allocated in this or upper scopes
            except KeyError:  # Array not allocated yet
                pass
            function_stream.write("%s *%s;\n" % (nodedesc.dtype.ctype, name))
            print("allocating %s" % node.data)
            self._frame._initcode.write(
                "%s = new %s DACE_ALIGN(64)[%s];\n"
                % (name, nodedesc.dtype.ctype, sym2cpp(nodedesc.total_size)),
                sdfg,
                state_id,
                node,
            )
            self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
            self.allocated_symbols.add(name)
        elif nodedesc.storage == dtypes.StorageType.CPU_Stack:
            # Check if array is already allocated
            try:
                self._dispatcher.defined_vars.get(name)
                return  # Array was already allocated in this or upper scopes
            except KeyError:  # Array not allocated yet
                pass
            function_stream.write(
                "%s *%s;\n#pragma omp threadprivate(%s)\n" % (nodedesc.dtype.ctype, name, name)
            )
            print("allocating %s" % node.data)

            tile_shapes = dict(
                I=nodedesc.tile_size[0], J=nodedesc.tile_size[1], K=dace.symbol("K")
            )
            subs = {
                (a if isinstance(a, dace.symbol) else a.args[1]): tile_shapes[
                    str(a if isinstance(a, dace.symbol) else a.args[1])
                ]
                for a in nodedesc.total_size.args
            }
            total_size = nodedesc.total_size.subs(subs)
            self._frame._initcode.write(
                """#pragma omp parallel
{
    %s = new %s DACE_ALIGN(64)[%s];
}
"""
                % (name, nodedesc.dtype.ctype, sym2cpp(total_size)),
                sdfg,
                state_id,
                node,
            )
            self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
            self.allocated_symbols.add(name)
        else:
            super().allocate_array(sdfg, dfg, state_id, node, function_stream, callsite_stream)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):

        name = node.data
        nodedesc = node.desc(sdfg)

        if nodedesc.transient is False:
            return

        if nodedesc.storage == dtypes.StorageType.CPU_Heap:
            if name in self.allocated_symbols:
                print("deleting %s" % node.data)
                self._frame._exitcode.write("delete[] %s;\n" % node.data, sdfg, state_id, node)
                self.allocated_symbols.remove(name)
        elif nodedesc.storage == dtypes.StorageType.CPU_Stack:
            if name in self.allocated_symbols:
                print("deleting %s" % node.data)
                self._frame._exitcode.write(
                    """#pragma omp parallel
{
    delete[] %s;
}
"""
                    % node.data,
                    sdfg,
                    state_id,
                    node,
                )
                self.allocated_symbols.remove(name)
        else:
            super().deallocate_array(sdfg, dfg, state_id, node, function_stream, callsite_stream)


def local_name(name, offset, is_target):
    CONFIG = dict(local_name_prefix="_gt_loc__", local_name_prefix_out="_gt_loc_out__")
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
            self.sdfg.instrument = dace.InstrumentationType.Timer

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

        def _make_mapped_computation(self, node: gt_ir.ApplyBlock, map_range):
            state = self.sdfg.add_state()

            tasklet = state.add_tasklet(
                name=self.new_tasklet_name(),
                inputs=node.mapped_input_memlets,
                outputs=node.mapped_output_memlets,
                code=node.tasklet_code,
            )
            map_entry, map_exit = state.add_map(name=self.new_map_name(), ndrange=map_range)

            for memlet_info in node.mapped_input_memlet_infos.values():
                name = memlet_info.outer_name
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

            return state

        def _make_parallel_computation(self, node: gt_ir.ApplyBlock):
            map_range = OrderedDict(
                i="{:d}:I{:+d}".format(*node.i_range),
                j="{:d}:J{:+d}".format(*node.j_range),
                k="{}:{}".format(*node.k_range),
            )
            state = self._make_mapped_computation(node, map_range)
            return state, state

        def _make_forward_computation(self, node: gt_ir.ApplyBlock):
            map_range = dict(
                i="{:d}:I{:+d}".format(*node.i_range),
                j="{:d}:J{:+d}".format(*node.j_range),  # _null_="0:1"
            )
            state = self._make_mapped_computation(node, map_range)
            loop_start, _, loop_end = self.sdfg.add_loop(
                None, state, None, "k", str(node.k_range[0]), f"k<{node.k_range[1]}", "k+1"
            )

            return loop_start, loop_end

        def _make_backward_computation(self, node: gt_ir.ApplyBlock):
            map_range = dict(
                i="{:d}:I{:+d}".format(*node.i_range),
                j="{:d}:J{:+d}".format(*node.j_range),  # _null_="0:1"
            )
            state = self._make_mapped_computation(node, map_range)
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
                if field.name in node.unreferenced:
                    continue
                shape = [
                    d + f
                    for d, f in zip(
                        self.symbolic_domain, node.fields_extents[field.name].frame_size
                    )
                ]
                if field.name in node.arg_fields:
                    I = dace.symbol(f"_{field.name}_I")
                    J = dace.symbol(f"_{field.name}_J")
                    K = dace.symbol(f"_{field.name}_K")
                    self.sdfg.add_array(
                        field.name,
                        strides=(J * K, K, 1),
                        shape=shape,
                        dtype=field.data_type.dtype.type,
                    )
                else:
                    assert field.name in node.temporary_fields
                    self.sdfg.add_transient(
                        # field.name, shape=shape, dtype=field.data_type.dtype.type, toplevel=True
                        field.name,
                        shape=shape,
                        dtype=field.data_type.dtype.type,
                    )
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


class DacePyModuleGenerator(gt_backend.BaseModuleGenerator):
    def __init__(self, backend_class, options):
        super().__init__(backend_class, options)

    def generate_imports(self):
        source = f"""
import functools
import ctypes

from gt4py.backend.dace_backend import load_dace_program

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
        for arg in run_args_names:
            if arg not in self.implementation_ir.unreferenced:
                if arg in self.implementation_ir.arg_fields:
                    run_args_strs.append(f"ctypes.c_void_p({self.generate_field_ptr_str(arg)})")
                else:
                    run_args_strs.append(
                        # "np.ctypeslib.as_ctypes(np.{dtype}({par_name}))".format(
                        "ctypes.{ctype_name}(np.{numpy_type}({par_name}))".format(
                            ctype_name=self.implementation_ir.parameters[arg].data_type.ctypes_str,
                            numpy_type=self.implementation_ir.parameters[arg].data_type.dtype.name,
                            dtype=self.implementation_ir.parameters[arg].data_type.dtype.name,
                            par_name=arg,
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
        source = (
            (
                "\nI=np.int32(_domain_[0])\nJ=np.int32(_domain_[1])\nK=np.int32(_domain_[2])\n"
                + total_field_sizes
                + "\n"
                + "\n".join(field_slices)
                + """

dace_lib['__dace_init_{program_name}']({run_args}, {total_field_sizes})
if exec_info is not None:
    exec_info['pyext_program_start_time'] = time.perf_counter()
dace_lib['__program_{program_name}']({run_args}, {total_field_sizes})
if exec_info is not None:
    exec_info['pyext_program_end_time'] = time.perf_counter()
dace_lib['__dace_exit_{program_name}']({run_args}, {total_field_sizes})
""".format(
                    program_name=self.implementation_ir.sdfg.name,
                    run_args=", ".join(run_args_strs),
                    total_field_sizes=",".join(v for k, v in sorted(symbol_ctype_strs.items())),
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

        source = source + (
            """
if exec_info is not None:
    exec_info["run_end_time"] = time.perf_counter()
"""
        )
        sources.extend(source.splitlines())

        return sources.text


class CPUDacePyModuleGenerator(DacePyModuleGenerator):
    def generate_field_ptr_str(self, arg):
        return f"{arg}.ctypes.data"


class GPUDacePyModuleGenerator(DacePyModuleGenerator):
    def generate_field_ptr_str(self, arg):
        return f"{arg}.data.data.ptr"


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
        # sdfg = performance_ir.sdfg

        import json

        from dace.transformation.dataflow.merge_arrays import MergeArrays

        sdfg.apply_transformations_repeated(MergeArrays)

        cls.transform_to_device(sdfg)
        cls.transform_optimize(sdfg)

        sdfg.save("tmp.sdfg")
        sdfg.validate()

        implementation_ir.sdfg = sdfg
        dace_build_path = os.path.relpath(cls.get_dace_module_path(stencil_id))
        os.makedirs(dace_build_path, exist_ok=True)

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


def global_ij_tiling(sdfg, tile_size=(8, 8)):
    input_arrays = dict()
    output_arrays = dict()
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.graph.nodes.AccessNode):
                if (
                    node.access is dace.AccessType.ReadOnly
                    or node.access is dace.AccessType.ReadWrite
                ) and not sdfg.arrays[node.data].transient:
                    num_accesses = input_arrays.get(node.data, 0)
                    input_arrays[node.data] = num_accesses + sum(
                        [e.data.num_accesses for e in state.out_edges(node)]
                    )

                if (
                    node.access is dace.AccessType.WriteOnly
                    or node.access is dace.AccessType.ReadWrite
                ) and not sdfg.arrays[node.data].transient:
                    num_accesses = output_arrays.get(node.data, 0)
                    output_arrays[node.data] = num_accesses + sum(
                        [e.data.num_accesses for e in state.in_edges(node)]
                    )

    # nest state
    import copy

    tmp_sdfg = copy.deepcopy(sdfg)
    for s in sdfg.nodes():
        sdfg.remove_node(s)
    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(
        tmp_sdfg, sdfg, list(input_arrays.keys()), list(output_arrays.keys())
    )
    nsdfg_node.symbol_mapping.update(
        # I=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[0]}, I-tile_i*{tile_size[0]})"),
        # J=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[1]}, J-tile_j*{tile_size[1]})"),
        I=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[0]}, I-tile_i)"),
        J=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[1]}, J-tile_j)"),
    )
    # map
    map_entry, map_exit = state.add_map(
        "global_tiling",
        ndrange=dict(
            # tile_i=f"0:int_ceil(I, {tile_size[0]})", tile_j=f"0:int_ceil(J, {tile_size[1]})"
            tile_i=f"0:I:{tile_size[0]}",
            tile_j=f"0:J:{tile_size[1]}",
        ),
    )
    map_entry.map.collapse = 2

    # conn_id = 0
    for array_name, num_accesses in input_arrays.items():
        array = sdfg.arrays[array_name]

        if not array.transient:
            map_entry.add_in_connector("IN_" + array_name)
            map_entry.add_out_connector("OUT_" + array_name)

            state.add_edge(
                state.add_read(array_name),
                None,
                map_entry,
                "IN_" + array_name,
                # f"IN_{conn_id}",
                memlet=dace.Memlet.simple(
                    array_name,
                    subset_str=",".join(
                        [f"0:{limit}" if str(limit) != "1" else "0" for limit in array.shape]
                    ),
                    num_accesses=num_accesses,
                ),
            )
            from dace.data import Scalar

            if isinstance(array, dace.data.Scalar):
                subset_str = "0"
            else:
                frame_i = dace.symbolic.pystr_to_symbolic(str(array.shape[0]) + "-I")
                frame_j = dace.symbolic.pystr_to_symbolic(str(array.shape[1]) + "-J")
                subset_str = ",".join(
                    [
                        # f"{tile_size[0]}*tile_i:Min({tile_size[0]}*(tile_i+1),I)+{frame_i}",
                        # f"{tile_size[1]}*tile_j:Min({tile_size[1]}*(tile_j+1),J)+{frame_j}",
                        f"tile_i:Min(tile_i+{tile_size[0]},I)+{frame_i}",
                        f"tile_j:Min(tile_j+{tile_size[1]},J)+{frame_j}",
                        f"0:{array.shape[2]}",
                    ]
                )

            state.add_edge(
                map_entry,
                "OUT_" + array_name,
                nsdfg_node,
                array_name,
                memlet=dace.Memlet.simple(
                    array_name, subset_str=subset_str, num_accesses=num_accesses
                ),
            )
        # conn_id += 1
    # conn_id = 0
    for array_name, num_accesses in output_arrays.items():
        array = sdfg.arrays[array_name]

        if not array.transient:
            map_exit.add_in_connector("IN_" + array_name)
            map_exit.add_out_connector("OUT_" + array_name)
            state.add_edge(
                map_exit,
                "OUT_" + array_name,
                state.add_write(array_name),
                None,
                memlet=dace.Memlet.simple(
                    array_name,
                    subset_str=",".join(
                        [f"0:{limit}" if str(limit) != "1" else "0" for limit in array.shape]
                    ),
                    num_accesses=num_accesses,
                ),
            )
            from dace.data import Scalar

            if isinstance(array, dace.data.Scalar):
                subset_str = "0"
            else:
                frame_i = dace.symbolic.pystr_to_symbolic(str(array.shape[0]) + "-I")
                frame_j = dace.symbolic.pystr_to_symbolic(str(array.shape[1]) + "-J")
                subset_str = ",".join(
                    [
                        # f"{tile_size[0]}*tile_i:Min({tile_size[0]+1}*tile_i,I)+{frame_i}",
                        # f"{tile_size[1]}*tile_j:Min({tile_size[1]+1}*tile_j,J)+{frame_j}",
                        f"tile_i:Min(tile_i+{tile_size[0]},I)+{frame_i}",
                        f"tile_j:Min(tile_j+{tile_size[1]},J)+{frame_j}",
                        f"0:{array.shape[2]}",
                    ]
                )

            state.add_edge(
                nsdfg_node,
                array_name,
                map_exit,
                "IN_" + array_name,
                memlet=dace.Memlet.simple(
                    array_name, subset_str=subset_str, num_accesses=num_accesses
                ),
            )

    if len(input_arrays) == 0:
        state.add_edge(map_entry, None, nsdfg_node, None, dace.EmptyMemlet())
    if len(output_arrays) == 0:
        state.add_edge(nsdfg_node, None, map_exit, None, dace.EmptyMemlet())

    # dace.dtypes.StorageType.register("CPU_Threadprivate_Persistent")
    import sympy

    # symbols = dict(_tile_I=dace.symbol("_tile_I"), _tile_J=dace.symbol("_tile_J"))
    # symbols['_tile_I'].set(tile_size[0])
    # symbols['_tile_J'].set(tile_size[1])

    # tile_sizes = dict(I=tile_size[0], J=tile_size[1], K="K")
    for array_name, array in nsdfg_node.sdfg.arrays.items():
        if array.transient:
            # array.shape = [
            #     f"{tile_sizes[str(s)]}"
            #     if isinstance(s, dace.symbolic.symbol)
            #     else s.subs({a: tile_sizes[str(a)] for a in s.args if str(a) in "IJ"})
            #     for s in array.shape
            # ]
            array.tile_size = tile_size
    print()
    #         array.storage = dace.dtypes.StorageType.CPU_Threadprivate_Persistent


import dace.transformation.pattern_matching as pattern_matching
from dace.properties import make_properties, Property, ShapeProperty
from dace.graph import nodes, nxutil


@registry.autoregister_params(singlestate=True)
@make_properties
class GlobalIJMapTiling(pattern_matching.Transformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    # Properties
    prefix = Property(dtype=str, default="tile", desc="Prefix for new range symbols")
    tile_size = ShapeProperty(dtype=tuple, default=(8, 8), desc="Tile size per dimension")

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(GlobalIJMapTiling._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # applies to IJ maps fully containing [0:I]x[0:J]
        map_entry = graph.nodes()[candidate[GlobalIJMapTiling._map_entry]]
        if not sorted(map_entry.map.params) == ["i", "j"]:
            return False
        free_symbols = map_entry.range.free_symbols
        if not sorted(free_symbols.keys()) == ["I", "J"]:
            return False
        I_symbol = free_symbols["I"]
        J_symbol = free_symbols["J"]
        range = map_entry.range.ranges
        if not range[0][0] <= 0 or not range[1][0] <= 0:
            return False
        if not range[0][1] - I_symbol <= 0 or not range[1][1] - J_symbol <= 0:
            return False
        if not range[0][2] == 1 or not range[1][2] == 1:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[GlobalIJMapTiling._map_entry]]
        return map_entry.map.label + ": " + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[GlobalIJMapTiling._map_entry]]

        # nd_to = symbolic.pystr_to_symbolic(
        #     f"int_ceil(I + 1, {tile_stride}) - 1"
        # )

        free_symbols = map_entry.range.free_symbols
        I_symbol = free_symbols["I"]
        J_symbol = free_symbols["J"]

        ranges = map_entry.range.ranges
        halo = (
            (ranges[0][0], ranges[0][1] - I_symbol + 1),
            (ranges[1][0], ranges[1][1] - J_symbol + 1),
        )

        from dace.transformation.dataflow.tiling import MapTiling

        maptiling = MapTiling(
            sdfg.sdfg_list.index(sdfg),
            self.state_id,
            {MapTiling._map_entry: self.subgraph[GlobalIJMapTiling._map_entry]},
            self.expr_index,
        )
        assert halo[0][0] <= 0
        assert halo[0][1] >= 0
        assert halo[1][0] <= 0
        assert halo[1][1] >= 0
        maptiling.tile_sizes = [t - h_l + h_r for t, (h_l, h_r) in zip(self.tile_size, halo)]
        maptiling.strides = self.tile_size
        maptiling.apply(sdfg)

        new_map_entry = graph.in_edges(map_entry)[0].src
        import dace.symbolic as symbolic

        new_map_entry.range.ranges[0] = (
            0,
            symbolic.pystr_to_symbolic(f"int_ceil(I, {self.tile_size[0]}) - 1"),
            1,
        )
        new_map_entry.range.ranges[1] = (
            0,
            symbolic.pystr_to_symbolic(f"int_ceil(J, {self.tile_size[1]}) - 1"),
            1,
        )


@registry.autoregister_params(singlestate=True)
@make_properties
class TaskletAsKLoop(pattern_matching.Transformation):
    """ Docstring TODO
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _tasklet = nodes.Tasklet("")
    _map_exit = nodes.MapExit(nodes.Map("", [], []))

    # Properties
    init = Property(default=0, desc="initial value for k")
    condition = Property(default="k<K", desc="stopping condition for the loop")
    step = Property(default="k+1", desc="value assigned to k every step (e.g. increment k+1)")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(
                TaskletAsKLoop._map_entry, TaskletAsKLoop._tasklet, TaskletAsKLoop._map_exit
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    def _k_range(self):
        if "<" in self.condition:
            k_min = self.init
            _, k_max = self.condition.split("<")
            k_max = k_max + " - 1"
        else:
            k_max = str(self.init)
            _, k_min = self.condition.split(">=")
        return k_min, k_max

    def apply(self, sdfg):
        graph: dace.sdfg.SDFGState = sdfg.nodes()[self.state_id]
        map_entry: dace.nodes.MapEntry = graph.nodes()[self.subgraph[TaskletAsKLoop._map_entry]]
        tasklet: dace.nodes.Tasklet = graph.nodes()[self.subgraph[TaskletAsKLoop._tasklet]]
        map_exit: dace.nodes.MapExit = graph.nodes()[self.subgraph[TaskletAsKLoop._map_exit]]
        from dace.transformation.helpers import nest_state_subgraph

        k_min, k_max = self._k_range()
        # fix outer edges to ij map
        import sympy

        k_symbol = dace.symbolic.symbol("k")
        for e in graph.in_edges(map_entry) + graph.out_edges(map_exit):
            for i, r in enumerate(e.data.subset.ranges):
                e.data.subset.ranges[i] = (
                    r[0].subs(dace.symbolic.symbol("k"), k_min),
                    r[1].subs(dace.symbolic.symbol("k"), k_max),
                    r[2],
                )

        # node = nest_state_subgraph(sdfg, graph, dace.sdfg.ScopeSubgraphView(graph, [tasklet]))
        nsdfg: SDFG = dace.SDFG(f"nested_k_loop_{graph.name}")
        nstate = nsdfg.add_state()
        nstate.add_nodes_from([tasklet])
        # nsdfg.add_nodes_from(dace.sdfg.ScopeSubgraphView(graph, [nstate]))

        in_prefix = f"__in_"
        out_prefix = f"__out_"

        nsdfg_in_arrays = set()
        for e in graph.out_edges(map_entry):
            nsdfg_in_arrays.add(in_prefix + e.data.data)
        nsdfg_out_arrays = set()
        for e in graph.in_edges(map_exit):
            nsdfg_out_arrays.add(out_prefix + e.data.data)

        for name in set(
            n.data
            for n in graph.nodes()
            if isinstance(n, dace.nodes.AccessNode) and n.access == dace.dtypes.AccessType.ReadOnly
        ):
            nsdfg.add_datadesc(in_prefix + name, sdfg.arrays[name])
        for name in set(
            n.data
            for n in graph.nodes()
            if isinstance(n, dace.nodes.AccessNode)
            and n.access == dace.dtypes.AccessType.WriteOnly
        ):
            nsdfg.add_datadesc(out_prefix + name, sdfg.arrays[name])

        read_accessors = dict()
        for name in nsdfg_in_arrays:
            read_accessors[name] = nstate.add_read(name)
        write_accessors = dict()
        for name in nsdfg_out_arrays:
            write_accessors[name] = nstate.add_write(name)

        for e in graph.out_edges(map_entry):
            nstate.add_edge(
                read_accessors[in_prefix + e.data.data],
                None,
                tasklet,
                e.dst_conn,
                memlet=dace.Memlet.simple(
                    in_prefix + e.data.data,
                    subset_str=str(e.data.subset),
                    num_accesses=e.data.num_accesses,
                ),
            )
        for e in graph.in_edges(map_exit):
            nstate.add_edge(
                tasklet,
                e.src_conn,
                write_accessors[out_prefix + e.data.data],
                None,
                memlet=dace.Memlet.simple(
                    out_prefix + e.data.data,
                    subset_str=str(e.data.subset),
                    num_accesses=e.data.num_accesses,
                ),
            )

        node = graph.add_nested_sdfg(nsdfg, sdfg, nsdfg_in_arrays, nsdfg_out_arrays)
        nstate = nsdfg.nodes()[0]

        conn_map_entry_to_nsdfg = dict()
        subsets_map_entry_to_nsdfg = dict()
        num_map_entry_to_nsdfg = dict()
        for e in graph.out_edges(map_entry):
            conn_map_entry_to_nsdfg[e.src_conn] = e.data.data

            subset = subsets_map_entry_to_nsdfg.get(e.data.data, e.data.subset)
            num = num_map_entry_to_nsdfg.get(e.data.data, e.data.num_accesses)
            for i, r in enumerate(subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        min(subset.ranges[i][0], e.data.subset[i][0]),
                        max(subset.ranges[i][1], e.data.subset[i][1]),
                        1,
                    )
                elif "k" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        0,
                        dace.symbolic.pystr_to_symbolic("K-1"),
                        1,
                    )  # graph.edges_between(
                    #     [
                    #         n
                    #         for n in graph.nodes()
                    #         if isinstance(n, dace.nodes.AccessNode)
                    #         and n.access == dace.AccessType.ReadOnly
                    #         and n.data == e.data.data
                    #     ][0],
                    #     map_entry,
                    # )[0].data.subset.ranges[i]
                subsets_map_entry_to_nsdfg[e.data.data] = subset
                num_map_entry_to_nsdfg[e.data.data] = num + e.data.num_accesses

        conn_map_exit_to_nsdfg = dict()
        for e in graph.in_edges(map_exit):
            conn_map_exit_to_nsdfg[e.dst_conn] = e.data.data

        for conn in map_entry.out_connectors:
            data_name = conn_map_entry_to_nsdfg[conn]
            graph.add_edge(
                map_entry,
                conn,
                node,
                in_prefix + conn_map_entry_to_nsdfg[conn],
                memlet=dace.Memlet.simple(
                    data=data_name,
                    subset_str=str(subsets_map_entry_to_nsdfg[data_name]),
                    num_accesses=num_map_entry_to_nsdfg[data_name],
                ),
            )

        conn_nsdfg_to_map_exit = dict()
        subsets_nsdfg_to_map_exit = dict()
        num_nsdfg_to_map_exit = dict()
        for e in graph.in_edges(map_exit):
            conn_nsdfg_to_map_exit[e.dst_conn] = e.data.data

            subset = subsets_nsdfg_to_map_exit.get(e.data.data, e.data.subset)
            num = num_nsdfg_to_map_exit.get(e.data.data, e.data.num_accesses)
            for i, r in enumerate(subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        min(subset.ranges[i][0], e.data.subset[i][0]),
                        max(subset.ranges[i][1], e.data.subset[i][1]),
                        1,
                    )
                elif "k" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        0,
                        dace.symbolic.pystr_to_symbolic("K-1"),
                        1,
                    )  # graph.edges_between(
                    #     map_exit,
                    #     [
                    #         n
                    #         for n in graph.nodes()
                    #         if isinstance(n, dace.nodes.AccessNode)
                    #         and n.access == dace.AccessType.WriteOnly
                    #         and n.data == e.data.data
                    #     ][0],
                    # )[0].data.subset.ranges[i]
                subsets_nsdfg_to_map_exit[e.data.data] = subset
                num_nsdfg_to_map_exit[e.data.data] = num + e.data.num_accesses
        for conn in map_exit.in_connectors:
            data_name = conn_nsdfg_to_map_exit[conn]
            graph.add_edge(
                node,
                out_prefix + conn_map_exit_to_nsdfg[conn],
                map_exit,
                conn,
                memlet=dace.Memlet.simple(
                    data=data_name,
                    subset_str=str(subsets_nsdfg_to_map_exit[data_name]),
                    num_accesses=num_nsdfg_to_map_exit[data_name],
                ),
            )
        for e in graph.in_edges(map_entry) + graph.out_edges(map_exit):
            if len(e.data.subset.ranges) >= 3 and "k" in dace.symbolic.symlist(
                e.data.subset.ranges[2]
            ):
                e.data.subset.ranges[2] = (0, dace.symbolic.pystr_to_symbolic("K-1"), 1)

        for e in nstate.in_edges(tasklet):
            outer_subset = subsets_map_entry_to_nsdfg[e.data.data[len(in_prefix) :]]
            for i, r in enumerate(e.data.subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    e.data.subset.ranges[i] = (
                        r[0] - outer_subset.ranges[i][0],
                        r[1] - outer_subset.ranges[i][0],
                        1,
                    )

        for e in nstate.out_edges(tasklet):
            outer_subset = subsets_nsdfg_to_map_exit[e.data.data[len(out_prefix) :]]
            for i, r in enumerate(e.data.subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    e.data.subset.ranges[i] = (
                        r[0] - outer_subset.ranges[i][0],
                        r[1] - outer_subset.ranges[i][0],
                        1,
                    )

        # Create a loop inside the nested SDFG
        nsdfg.add_loop(None, nstate, None, "k", self.init, self.condition, self.step)
        graph.remove_node(tasklet)
        # outer_in_edges = {e.dst_conn: e for e in graph.in_edges(node)}
        # outer_out_edges = {e.src_conn: e for e in graph.out_edges(node)}
        #
        # for e in nstate.in_edges(tasklet):
        #     assert all(r == (0, 0, 1) for r in e.data.subset.ranges)
        #     assert e.src.data in outer_in_edges
        #     outer_edge = outer_in_edges[e.src.data]
        #     for i, r in enumerate(outer_edge.data.subset.ranges):
        #         e.data.subset.ranges[i] = r
        #
        # for e in nstate.out_edges(tasklet):
        #     assert all(r == (0, 0, 1) for r in e.data.subset.ranges)
        #     assert e.dst.data in outer_out_edges
        #     outer_edge = outer_out_edges[e.dst.data]
        #     for i, r in enumerate(outer_edge.data.subset.ranges):
        #         e.data.subset.ranges[i] = r

        #     e.data.subset.ranges[i] = r
        # if len(e.data.subset.ranges) > 2:
        #     e.data.subset.ranges[2] = (
        #         dace.symbolic.pystr_to_symbolic("k"),
        #         dace.symbolic.pystr_to_symbolic("k"),
        #         dace.symbolic.pystr_to_symbolic("1"),
        #     )


def eliminate_trivial_k_loop(sdfg: dace.SDFG, state: dace.SDFGState):
    sdfg.predecessor_states(state)
    if not len(sdfg.successors(state)) == 2:
        return
    if not len(sdfg.predecessors(state)) == 2:
        return
    init, condition, step = None, None, None
    for s in sdfg.predecessors(state):
        edges = sdfg.edges_between(s, state)
        if not len(edges) == 1:
            return
        if edges[0].data.condition.as_string == "" and s in sdfg.predecessor_states(state):
            init = edges[0].data.assignments["k"]
            init_state = s
        elif not edges[0].data.condition.as_string == "":
            return
        else:
            step = edges[0].data.assignments["k"]
            loop_end_state = s
    for s in sdfg.successors(state):
        edges = sdfg.edges_between(state, s)
        if edges:
            if not len(edges) == 1:
                return
            if not edges[0].data.condition.as_string == "":
                condition = edges[0].data.condition
                loop_start_state = s
            else:
                exit_state = s

    if "<" in condition.as_string:
        k_min = init
        _, k_max = condition.as_string.split("<")
        k_max = k_max + " - 1"
    else:
        k_max = str(init)
        _, k_min = condition.as_string.split(">=")

    if not dace.symbolic.pystr_to_symbolic(f"({k_min})-({k_max})") == 0:
        return

    # add edge from pred directly to loop states
    sdfg.add_edge(init_state, loop_start_state, dace.InterstateEdge(assignments={"k": init}))
    # add edge from loop states directly to succ
    sdfg.add_edge(loop_end_state, exit_state, dace.InterstateEdge())
    # remove guard & edges involving guard
    for s in sdfg.successors(state):
        for edge in sdfg.edges_between(state, s):
            sdfg.remove_edge(edge)
    for s in sdfg.predecessors(state):
        for edge in sdfg.edges_between(s, state):
            sdfg.remove_edge(edge)
    sdfg.remove_node(state)


def outer_k_loop_to_inner_map(sdfg: dace.SDFG, state: dace.SDFGState):
    sdfg.predecessor_states(state)
    if not len(sdfg.successors(state)) == 2:
        return
    if not len(sdfg.predecessors(state)) == 2:
        return
    init, condition, step = None, None, None
    for s in sdfg.predecessors(state):
        edges = sdfg.edges_between(s, state)
        if not len(edges) == 1:
            return
        if edges[0].data.condition.as_string == "" and s in sdfg.predecessor_states(state):
            init = edges[0].data.assignments["k"]
            init_state = s
        elif not edges[0].data.condition.as_string == "":
            return
        else:
            step = edges[0].data.assignments["k"]
            loop_end_state = s
    for s in sdfg.successors(state):
        edges = sdfg.edges_between(state, s)
        if edges:
            if not len(edges) == 1:
                return
            if not edges[0].data.condition.as_string == "":
                condition = edges[0].data.condition
                loop_start_state = s
            else:
                exit_state = s
    # for state in loop...
    loop_states = []
    s = loop_start_state
    while s is not state:
        if not len(sdfg.successors(s)) == 1:
            return
        else:
            loop_states.append(s)
            s = sdfg.successors(s)[0]
    assert loop_end_state is loop_states[-1]

    # replace tasklet with nestedsdfg
    for s in loop_states:
        sdfg.apply_transformations(
            TaskletAsKLoop,
            states=[s],
            validate=False,
            options=dict(init=init, step=step, condition=condition.as_string),
        )
    # add edge from pred directly to loop states
    sdfg.add_edge(init_state, loop_start_state, dace.InterstateEdge())
    # add edge from loop states directly to succ
    sdfg.add_edge(loop_end_state, exit_state, dace.InterstateEdge())
    # remove guard & edges involving guard
    for s in sdfg.successors(state):
        for edge in sdfg.edges_between(state, s):
            sdfg.remove_edge(edge)
    for s in sdfg.predecessors(state):
        for edge in sdfg.edges_between(s, state):
            sdfg.remove_edge(edge)
    sdfg.remove_node(state)


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
        # from dace.transformation.dataflow import MergeArrays
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
        # sdfg.apply_strict_transformations(validate=False)
        # sdfg.apply_transformations_repeated([MapFusion], validate=False)
        # # from dace.transformation.dataflow import Vectorization
        #
        # #
        # # sdfg.apply_transformations_repeated(Vectorization)
        # global_ij_tiling(sdfg, tile_size=(8, 8))
        pass


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

        sdfg.apply_transformations([GPUTransformSDFG], validate=False)

    @classmethod
    def transform_optimize(cls, sdfg):
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
        pass
