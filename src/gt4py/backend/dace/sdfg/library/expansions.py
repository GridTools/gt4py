import dace
import dace.subsets
from dace import data as dace_data

from gt4py import ir as gt_ir


@dace.library.expansion
class ForLoopExpandTransformation(dace.library.ExpandTransformation):

    environments = []

    # def _make_parallel_computation(self, node: gt_ir.ApplyBlock):
    #     map_range = OrderedDict(
    #         i="{:d}:I{:+d}".format(*node.i_range),
    #         j="{:d}:J{:+d}".format(*node.j_range),
    #         k="{}:{}".format(*node.k_range),
    #     )
    #     state = self._make_mapped_computation(node, map_range)
    #     return state, state
    #
    # def _make_forward_computation(self, node: gt_ir.ApplyBlock):
    #     map_range = dict(
    #         i="{:d}:I{:+d}".format(*node.i_range),
    #         j="{:d}:J{:+d}".format(*node.j_range),  # _null_="0:1"
    #     )
    #     state = self._make_mapped_computation(node, map_range)
    #     loop_start, _, loop_end = self.sdfg.add_loop(
    #         None, state, None, "k", str(node.k_range[0]), f"k<{node.k_range[1]}", "k+1"
    #     )
    #
    #     return loop_start, loop_end
    #
    # def _make_backward_computation(self, node: gt_ir.ApplyBlock):
    #     map_range = dict(
    #         i="{:d}:I{:+d}".format(*node.i_range),
    #         j="{:d}:J{:+d}".format(*node.j_range),  # _null_="0:1"
    #     )
    #     state = self._make_mapped_computation(node, map_range)
    #     loop_start, _, loop_end = self.sdfg.add_loop(
    #         None, state, None, "k", str(node.k_range[1]) + "-1", f"k>={node.k_range[0]}", "k-1"
    #     )
    #
    #     return loop_start, loop_end
    @classmethod
    def _map(cls, library_node, mapped_sdfg, not_yet_mapped_vars, variable):
        limit_var = variable.upper()
        iter_var = variable.lower()

        tmp_sdfg = dace.SDFG(library_node.label + f"_tmp_{limit_var}_sdfg")
        tmp_state = tmp_sdfg.add_state(library_node.label + f"_tmp_{limit_var}_state")

        symbol_mapping = {}
        for k in mapped_sdfg.free_symbols:
            symbol_mapping[k] = dace.symbolic.symbol(k, mapped_sdfg.symbols[k])
            tmp_sdfg.symbols[k] = mapped_sdfg.symbols[k]

        nsdfg_node = tmp_state.add_nested_sdfg(
            mapped_sdfg,
            tmp_sdfg,
            set("IN_" + inp for inp in library_node.inputs),
            set("OUT_" + outp for outp in library_node.outputs),
            symbol_mapping=symbol_mapping,
        )

        tmp_sdfg.save("tmp__map.sdfg")

        # ndrange = {iter_var: str(library_node.k_range) if limit_var == "K" else f"0:{limit_var}"}
        ndrange = {iter_var: f"0:_{limit_var}_loc"}
        map_entry, map_exit = tmp_state.add_map(
            library_node.label + f"_{variable}_map", ndrange=ndrange
        )
        for input in library_node.inputs:
            acc_num = (
                dace.DYNAMIC
                if all(
                    acc.num == dace.DYNAMIC
                    for acc in library_node.read_accesses.values()
                    if input == acc.outer_name
                )
                else 1
            )
            output = input
            is_array = isinstance(mapped_sdfg.arrays["IN_" + input], dace_data.Array)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.input_extents[input][i]

                if var in not_yet_mapped_vars:
                    # if var == "K":
                    #     subset = library_node.k_range.ranges[0]
                    #     subset_strs.append(
                    #         f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                    #     )
                    # else:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent+upper_extent)}")
                elif var == variable:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent+upper_extent)+1}"
                    )
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent + upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs) if is_array else "0"

            map_entry.add_out_connector("OUT_" + output)
            tmp_state.add_edge(
                map_entry,
                "OUT_" + output,
                nsdfg_node,
                "IN_" + input,
                dace.memlet.Memlet.simple(
                    data="IN_" + input, subset_str=subset_str, num_accesses=acc_num
                ),
            )
            mapped_shape = library_node.input_extents[input].shape
            # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"_I_loc+{mapped_shape[0]-1}",
                f"_J_loc+{mapped_shape[1]-1}",
                f"_K_loc+{mapped_shape[2]-1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )

            if is_array:
                tmp_sdfg.add_array(
                    "IN_" + input,
                    shape=tuple(
                        m if var not in not_yet_mapped_vars | {variable} else f
                        for var, m, f in zip("IJK", mapped_shape, full_shape)
                    ),
                    dtype=mapped_sdfg.arrays["IN_" + input].dtype,
                    strides=mapped_sdfg.arrays["IN_" + input].strides,
                    total_size=mapped_sdfg.arrays["IN_" + input].total_size,
                    storage=dace.StorageType.Default,
                )
            else:
                tmp_sdfg.add_scalar(
                    "IN_" + input,
                    dtype=mapped_sdfg.arrays["IN_" + input].dtype,
                    storage=dace.StorageType.Default,
                )
            map_entry.add_in_connector("IN_" + input)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.input_extents[input][i]

                if var in not_yet_mapped_vars | {variable.upper()}:
                    # if var == "K":
                    #     subset = library_node.k_range.ranges[0]
                    #     subset_strs.append(
                    #         f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                    #     )
                    # else:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent+upper_extent)}")
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent + upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs) if is_array else "0"

            tmp_state.add_edge(
                tmp_state.add_read("IN_" + input),
                None,
                map_entry,
                "IN_" + input,
                dace.memlet.Memlet.simple(
                    "IN_" + input, subset_str=subset_str, num_accesses=acc_num
                ),
            )
        for output in library_node.outputs:
            acc_num = (
                dace.DYNAMIC
                if all(
                    acc.num == dace.DYNAMIC
                    for acc in library_node.write_accesses.values()
                    if output == acc.outer_name
                )
                else 1
            )
            input = output
            is_array = isinstance(mapped_sdfg.arrays["OUT_" + output], dace_data.Array)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.output_extents[output][i]

                if var in not_yet_mapped_vars:
                    # if var == "K":
                    #     subset = library_node.k_range.ranges[0]
                    #     subset_strs.append(
                    #         f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                    #     )
                    # else:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent+upper_extent)}")
                elif var == variable:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent+upper_extent)+1}"
                    )
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent + upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs) if is_array else "0"

            map_exit.add_in_connector("IN_" + input)
            tmp_state.add_edge(
                nsdfg_node,
                "OUT_" + output,
                map_exit,
                "IN_" + input,
                dace.memlet.Memlet.simple(
                    data="OUT_" + output, subset_str=subset_str, num_accesses=acc_num
                ),
            )
            mapped_shape = library_node.output_extents[output].shape
            # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"_I_loc+{mapped_shape[0]-1}",
                f"_J_loc+{mapped_shape[1]-1}",
                f"_K_loc+{mapped_shape[2]-1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )
            if output not in tmp_sdfg.arrays:
                if is_array:
                    tmp_sdfg.add_array(
                        "OUT_" + output,
                        shape=tuple(
                            m if m in not_yet_mapped_vars else f
                            for m, f in zip(mapped_shape, full_shape)
                        )
                        if isinstance(mapped_sdfg.arrays["OUT_" + output], dace_data.Array)
                        else "1",
                        dtype=mapped_sdfg.arrays["OUT_" + output].dtype,
                        strides=mapped_sdfg.arrays["OUT_" + output].strides,
                        total_size=mapped_sdfg.arrays["OUT_" + output].total_size,
                        storage=dace.StorageType.Default,
                    )
                else:
                    tmp_sdfg.add_scalar(
                        "OUT_" + output,
                        dtype=mapped_sdfg.arrays["OUT_" + output].dtype,
                        storage=dace.StorageType.Default,
                    )
            map_exit.add_out_connector("OUT_" + output)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.output_extents[input][i]

                if var in not_yet_mapped_vars | {variable.upper()}:
                    # if var == "K":
                    #     subset = library_node.k_range.ranges[0]
                    #     subset_strs.append(
                    #         f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                    #     )
                    # else:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent+upper_extent)}")
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent+upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs) if is_array else "0"

            tmp_state.add_edge(
                map_exit,
                "OUT_" + output,
                tmp_state.add_write("OUT_" + output),
                None,
                dace.memlet.Memlet.simple(
                    "OUT_" + output, subset_str=subset_str, num_accesses=acc_num
                ),
            )

        if len(library_node.inputs) == 0:
            tmp_state.add_edge(map_entry, None, nsdfg_node, None, dace.EmptyMemlet())
        if len(library_node.outputs) == 0:
            tmp_state.add_edge(nsdfg_node, None, map_entry, None, dace.EmptyMemlet())

        from dace.transformation.interstate import InlineSDFG

        tmp_sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
        return tmp_sdfg

    @classmethod
    def _loop(cls, library_node, mapped_sdfg, not_yet_mapped_vars, variable):
        limit_var = variable.upper()
        iter_var = variable.lower()

        tmp_sdfg = dace.SDFG(library_node.label + f"_tmp_{limit_var}_sdfg")
        tmp_state = tmp_sdfg.add_state(library_node.label + f"_tmp_{limit_var}_state")

        symbol_mapping = {}
        for k in mapped_sdfg.free_symbols:
            symbol_mapping[k] = dace.symbolic.symbol(k, mapped_sdfg.symbols[k])
            tmp_sdfg.symbols[k] = mapped_sdfg.symbols[k]

        nsdfg_node = tmp_state.add_nested_sdfg(
            mapped_sdfg,
            tmp_sdfg,
            set("IN_" + inp for inp in library_node.inputs),
            set("OUT_" + outp for outp in library_node.outputs),
            symbol_mapping=symbol_mapping,
        )

        # ndrange = {iter_var: str(library_node.k_range) if limit_var == "K" else f"0:{limit_var}"}

        for input in library_node.inputs:
            acc_num = (
                dace.DYNAMIC
                if all(
                    acc.num == dace.DYNAMIC
                    for acc in library_node.read_accesses.values()
                    if input == acc.outer_name
                )
                else 1
            )
            is_array = isinstance(mapped_sdfg.arrays["IN_" + input], dace_data.Array)
            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.input_extents[input][i]

                if var in not_yet_mapped_vars:
                    # if var == "K":
                    #     subset = library_node.k_range.ranges[0]
                    #     subset_strs.append(
                    #         f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                    #     )
                    # else:
                    subset_strs.append(
                        f"0:{dace.symbolic.pystr_to_symbolic(f'({library_node.ranges[2][1]}) - ({library_node.ranges[2][0]})')}+{(-lower_extent+upper_extent)}"
                    )
                elif var == variable:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent+upper_extent)+1}"
                    )
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent + upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs) if is_array else "0"

            tmp_state.add_edge(
                tmp_state.add_read("IN_" + input),
                None,
                nsdfg_node,
                "IN_" + input,
                dace.memlet.Memlet.simple(
                    data="IN_" + input, subset_str=subset_str, num_accesses=acc_num
                ),
            )
            mapped_shape = library_node.input_extents[input].shape
            # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"_I_loc+{mapped_shape[0]-1}",
                f"_J_loc+{mapped_shape[1]-1}",
                f"_K_loc+{mapped_shape[2]-1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )
            if is_array:
                tmp_sdfg.add_array(
                    "IN_" + input,
                    shape=tuple(
                        m if var not in not_yet_mapped_vars | {variable} else f
                        for var, m, f in zip("IJK", mapped_shape, full_shape)
                    ),
                    dtype=mapped_sdfg.arrays["IN_" + input].dtype,
                    strides=mapped_sdfg.arrays["IN_" + input].strides,
                    total_size=mapped_sdfg.arrays["IN_" + input].total_size,
                    storage=dace.StorageType.Default,
                )
            else:
                tmp_sdfg.add_scalar(
                    "IN_" + input,
                    dtype=mapped_sdfg.arrays["IN_" + input].dtype,
                    storage=dace.StorageType.Default,
                )

        for output in library_node.outputs:
            acc_num = (
                dace.DYNAMIC
                if all(
                    acc.num == dace.DYNAMIC
                    for acc in library_node.write_accesses.values()
                    if output == acc.outer_name
                )
                else 1
            )
            is_array = isinstance(mapped_sdfg.arrays["OUT_" + output], dace_data.Array)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.output_extents[output][i]

                if var in not_yet_mapped_vars:
                    # if var == "K":
                    #     subset = library_node.k_range.ranges[0]
                    #     subset_strs.append(
                    #         f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                    #     )
                    # else:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent+upper_extent)}")
                elif var == variable:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent+upper_extent)+1}"
                    )
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent + upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs) if is_array else "0"

            tmp_state.add_edge(
                nsdfg_node,
                "OUT_" + output,
                tmp_state.add_write("OUT_" + output),
                None,
                dace.memlet.Memlet.simple(
                    data="OUT_" + output, subset_str=subset_str, num_accesses=acc_num
                ),
            )
            mapped_shape = library_node.output_extents[output].shape
            # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"_I_loc+{mapped_shape[0]-1}",
                f"_J_loc+{mapped_shape[1]-1}",
                f"_K_loc+{mapped_shape[2]-1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )
            if output not in tmp_sdfg.arrays:
                if is_array:
                    tmp_sdfg.add_array(
                        "OUT_" + output,
                        shape=tuple(
                            m if m in not_yet_mapped_vars else f
                            for m, f in zip(mapped_shape, full_shape)
                        )
                        if isinstance(mapped_sdfg.arrays["OUT_" + output], dace_data.Array)
                        else "1",
                        dtype=mapped_sdfg.arrays["OUT_" + output].dtype,
                        strides=mapped_sdfg.arrays["OUT_" + output].strides,
                        total_size=mapped_sdfg.arrays["OUT_" + output].total_size,
                        storage=dace.StorageType.Default,
                    )
                else:
                    tmp_sdfg.add_scalar(
                        "OUT_" + output,
                        dtype=mapped_sdfg.arrays["OUT_" + output].dtype,
                        storage=dace.StorageType.Default,
                    )

        if library_node.iteration_order == gt_ir.IterationOrder.BACKWARD:
            tmp_sdfg.add_loop(
                tmp_sdfg.add_state(f"_tmp_{limit_var}_init_state"),
                tmp_state,
                tmp_sdfg.add_state(f"_tmp_{limit_var}_end_state"),
                loop_var=iter_var,
                initialize_expr=f"_{limit_var}_loc-1",
                condition_expr=f"{iter_var}>=0",
                increment_expr=f"{iter_var}-1",
            )
        else:
            tmp_sdfg.add_loop(
                tmp_sdfg.add_state(f"_tmp_{limit_var}_init_state"),
                tmp_state,
                tmp_sdfg.add_state(f"_tmp_{limit_var}_end_state"),
                loop_var=iter_var,
                initialize_expr=f"0",
                condition_expr=f"{iter_var}<_{limit_var}_loc",
                increment_expr=f"{iter_var}+1",
            )

        from dace.transformation.interstate import InlineSDFG

        tmp_sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
        return tmp_sdfg

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        tmp_sdfg = dace.SDFG(node.label + "_tmp_tasklet_sdfg")
        tmp_state = tmp_sdfg.add_state(node.label + "_tmp_tasklet_state")
        tasklet = tmp_state.add_tasklet(
            name=node.name + "_tasklet",
            inputs=set(node.read_accesses.keys()),
            outputs=set(node.write_accesses.keys()),
            code=node.code.as_string,
        )
        read_accessors = dict()
        write_accessors = dict()

        for edge in parent_state.in_edges(node):
            parent_array = parent_sdfg.arrays[edge.data.data]
            if isinstance(parent_array, dace_data.Scalar):
                tmp_sdfg.add_scalar(
                    "IN_" + edge.data.data, parent_array.dtype, storage=dace.StorageType.Default
                )
            else:
                tmp_sdfg.add_array(
                    "IN_" + edge.data.data,
                    dtype=parent_array.dtype,
                    shape=node.input_extents[edge.data.data].shape,
                    strides=parent_array.strides,
                    # total_size=parent_array.total_size.subs(
                    #     {
                    #         I: I - (node.ranges[0][1] - node.ranges[0][0]),
                    #         J: J - (node.ranges[1][1] - node.ranges[1][0]),
                    #         K: dace.symbolic.pystr_to_symbolic(
                    #             f"{node.ranges[2][1]} - {node.ranges[2][0]}"
                    #         ),
                    #     }
                    # ),
                    total_size=parent_array.total_size,
                    storage=dace.StorageType.Default,
                )
        for edge in parent_state.out_edges(node):
            parent_array = parent_sdfg.arrays[edge.data.data]
            if isinstance(parent_array, dace_data.Scalar):
                tmp_sdfg.add_scalar(
                    "OUT_" + edge.data.data, parent_array.dtype, storage=dace.StorageType.Default
                )
            else:
                tmp_sdfg.add_array(
                    "OUT_" + edge.data.data,
                    dtype=parent_array.dtype,
                    shape=node.output_extents[edge.data.data].shape,
                    strides=parent_array.strides,
                    # total_size=parent_array.total_size.subs(
                    #                     {
                    #                         I: I - (node.ranges[0][1] - node.ranges[0][0]),
                    #                         J: J - (node.ranges[1][1] - node.ranges[1][0]),
                    #                         K: K
                    #                         - dace.symbolic.pystr_to_symbolic(
                    #                             f"{node.ranges[2][1]} - {node.ranges[2][0]}"
                    #                         ),
                    #                     }
                    #                 ),
                    total_size=parent_array.total_size,
                    storage=dace.StorageType.Default,
                )

        for name in set(acc.outer_name for acc in node.read_accesses.values()):
            read_accessors[name] = tmp_state.add_read("IN_" + name)
        for name in set(acc.outer_name for acc in node.write_accesses.values()):
            write_accessors[name] = tmp_state.add_write("OUT_" + name)
        for name, acc in node.read_accesses.items():

            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = (
                ",".join(
                    str(o - e)
                    for o, e in zip(offset_tuple, node.input_extents[acc.outer_name].lower_indices)
                )
                if isinstance(tmp_sdfg.arrays["IN_" + acc.outer_name], dace_data.Array)
                else "0"
            )
            tmp_state.add_edge(
                read_accessors[acc.outer_name],
                None,
                tasklet,
                name,
                dace.memlet.Memlet.simple(
                    "IN_" + acc.outer_name, subset_str=subset_str, num_accesses=acc.num
                ),
            )
        for name, acc in node.write_accesses.items():
            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = (
                ",".join(
                    str(o - e)
                    for o, e in zip(
                        offset_tuple, node.output_extents[acc.outer_name].lower_indices
                    )
                )
                if isinstance(tmp_sdfg.arrays["OUT_" + acc.outer_name], dace_data.Array)
                else "0"
            )
            tmp_state.add_edge(
                tasklet,
                name,
                write_accessors[acc.outer_name],
                None,
                dace.memlet.Memlet.simple(
                    "OUT_" + acc.outer_name, subset_str=subset_str, num_accesses=acc.num
                ),
            )
        #
        # for k in tmp_sdfg.symbols.keys():
        #     tmp_sdfg.symbols[k] = parent_sdfg.symbols[k]
        for k in tasklet.free_symbols:
            tmp_sdfg.symbols[k] = parent_sdfg.symbols[k]

        not_yet_mapped_vars = set()
        for variable in reversed(node.loop_order):
            if variable == "K" and node.iteration_order is not gt_ir.IterationOrder.PARALLEL:
                tmp_sdfg = cls._loop(
                    node, tmp_sdfg, not_yet_mapped_vars=not_yet_mapped_vars, variable=variable
                )
            else:
                tmp_sdfg = cls._map(
                    node, tmp_sdfg, not_yet_mapped_vars=not_yet_mapped_vars, variable=variable
                )
            not_yet_mapped_vars.add(variable)

        symbol_mapping = {}
        for k in tmp_sdfg.free_symbols:
            symbol_mapping[k] = k

        symbol_mapping.update(
            _I_loc=f"I+{node.ranges[0][1] - node.ranges[0][0]}",
            _J_loc=f"J+{node.ranges[1][1] - node.ranges[1][0]}",
            _K_loc=f"({node.ranges[2][1]}) - ({node.ranges[2][0]})",
        )

        k_loc = dace.symbolic.pystr_to_symbolic(f"({node.ranges[2][1]}) - ({node.ranges[2][0]})")

        # if len(k_loc.free_symbols) == 0:
        #     del symbol_mapping["_K_loc"]
        #     # tmp_sdfg.add_constant("_K_loc", dace.dtypes.int32(int(k_loc)), dace.dtypes.int32())
        #     tmp_sdfg.replace("_K_loc", str(k_loc))
        # else:
        #     symbol_mapping["_K_loc"] = k_loc

        from dace.transformation.interstate import InlineSDFG

        tmp_sdfg.apply_transformations_repeated(InlineSDFG, validate=False)

        from gt4py.backend.dace.sdfg.api import replace_recursive

        for k in "IJK":
            replace_recursive(tmp_sdfg, f"_{k}_loc", str(symbol_mapping[f"_{k}_loc"]))

        from gt4py.backend.dace.sdfg.transforms import RemoveTrivialLoop
        from gt4py.backend.dace.sdfg.api import apply_transformations_repeated_recursive
        from dace.transformation.interstate import EndStateElimination, StateAssignElimination
        from dace.transformation.dataflow import MapCollapse

        apply_transformations_repeated_recursive(
            tmp_sdfg,
            [RemoveTrivialLoop, EndStateElimination, StateAssignElimination, MapCollapse],
            validate=False,
        )
        tmp_sdfg.apply_strict_transformations(validate=False)
        dace.propagate_memlets_sdfg(tmp_sdfg)
        res = dace.nodes.NestedSDFG(
            label=node.label,
            sdfg=tmp_sdfg,
            inputs=node.in_connectors,
            outputs=node.out_connectors,
            symbol_mapping=symbol_mapping,
        )
        res.sdfg.parent = parent_state
        res.sdfg.parent_sdfg = parent_sdfg
        return res

        # res_sdfg = dace.SDFG(node.label + "_expanded_sdfg")
        # res_state = res_sdfg.add_state(node.label + "_expanded_state")
        # nsdfg = res_state.add_nested_sdfg(
        #     tmp_sdfg, res_sdfg, inputs=node.inputs, outputs=node.outputs
        # )
        # read_accessors = {}
        # write_accessors = {}
        # for name, array in tmp_sdfg.arrays.items():
        #     if not array.transient:
        #
        #         if name in node.inputs:
        #             read_accessors[name] = res_state.add_read("IN_" + name)
        #             if isinstance(array, dace_data.Scalar):
        #                 res_sdfg.add_scalar("IN_" + name, array.dtype, transient=array.transient)
        #             else:
        #                 assert isinstance(array, dace_data.Array)
        #                 res_sdfg.add_array(
        #                     "IN_" + name,
        #                     array.shape,
        #                     array.dtype,
        #                     transient=array.transient,
        #                     strides=array.strides,
        #                     total_size=array.total_size,
        #                 )
        #         if name in node.outputs:
        #             write_accessors[name] = res_state.add_write("OUT_" + name)
        #             if isinstance(array, dace_data.Scalar):
        #                 res_sdfg.add_scalar("OUT_" + name, array.dtype, transient=array.transient)
        #             else:
        #                 assert isinstance(array, dace_data.Array)
        #                 res_sdfg.add_array(
        #                     "OUT_" + name,
        #                     array.shape,
        #                     array.dtype,
        #                     transient=array.transient,
        #                     strides=array.strides,
        #                     total_size=array.total_size,
        #                 )
        #
        # for outer_edge in parent_state.in_edges(node):
        #     outer_memlet = outer_edge.data
        #     name = outer_memlet.data
        #     res_state.add_edge(
        #         read_accessors[name],
        #         None,
        #         nsdfg,
        #         "IN_"+name,
        #         dace.memlet.Memlet.simple(
        #             "IN_" + name,
        #             subset_str=",".join(f"0:{s}" for s in res_sdfg.arrays["IN_" + name].shape),
        #             num_accesses=outer_memlet.num_accesses,
        #         ),
        #     )
        # for outer_edge in parent_state.out_edges(node):
        #     outer_memlet = outer_edge.data
        #     name = outer_memlet.data
        #     res_state.add_edge(
        #         nsdfg,
        #         "OUT_"+name,
        #         write_accessors[name],
        #         None,
        #         dace.memlet.Memlet.simple(
        #             "OUT_" + name,
        #             subset_str=",".join(f"0:{s}" for s in res_sdfg.arrays["OUT_" + name].shape),
        #             num_accesses=outer_memlet.num_accesses,
        #         ),
        #     )

        # # from dace.transformation.interstate import InlineSDFG
        # res_sdfg.apply_strict_transformations()
        # tmp_sdfg.validate()
        # res_sdfg.save('res_sdfg.sdfg')
        # res_sdfg.validate()

        # return dace.nodes.NestedSDFG(
        #     label=node.label,
        #     sdfg=res_sdfg,
        #     inputs=node.in_connectors,
        #     outputs=node.out_connectors,
        #     symbol_mapping=symbol_mapping,
        # )
