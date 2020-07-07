import dace
import dace.subsets
from dace import data as dace_data
import dace.library
from gt4py import ir as gt_ir


class StencilExpander:
    def __init__(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, library_node):
        self.not_yet_mapped_variables = set()
        self.parent_sdfg = parent_sdfg
        self.parent_state = parent_state
        self.library_node = library_node

        self.inner_sdfg = None
        self.inner_state = None

    def _add_edges_map(
        self,
        sdfg,
        state,
        nsdfg_node,
        names,
        prefix,
        extents,
        map_scope_delimiter,
        access_dict,
        variable,
    ):
        mapped_sdfg = nsdfg_node.sdfg

        subsets = {}
        for name, extent in extents.items():
            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = extents[name][i]

                if var in self.not_yet_mapped_variables:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent + upper_extent)}")
                elif var == variable:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent + upper_extent) + 1}"
                    )
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent + upper_extent + 1}"
                    )
            subsets[name] = ",".join(subset_strs)

        for name in names:
            is_array = isinstance(mapped_sdfg.arrays[prefix + name], dace_data.Array)

            if isinstance(map_scope_delimiter, dace.nodes.MapEntry):
                map_scope_delimiter.add_out_connector("OUT_" + name)
                state.add_edge(
                    map_scope_delimiter,
                    "OUT_" + name,
                    nsdfg_node,
                    prefix + name,
                    dace.memlet.Memlet.simple(
                        data=prefix + name, subset_str=subsets[name], num_accesses=1
                    ),
                )
            else:
                map_scope_delimiter.add_in_connector("IN_" + name)
                state.add_edge(
                    nsdfg_node,
                    prefix + name,
                    map_scope_delimiter,
                    "IN_" + name,
                    dace.memlet.Memlet.simple(
                        data=prefix + name, subset_str=subsets[name], num_accesses=1
                    ),
                )
            mapped_shape = extents[name].shape
            # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"_I_loc+{mapped_shape[0] - 1}",
                f"_J_loc+{mapped_shape[1] - 1}",
                f"_K_loc+{mapped_shape[2] - 1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )
            if name not in sdfg.arrays:
                if is_array:
                    sdfg.add_array(
                        prefix + name,
                        shape=tuple(
                            m if m in self.not_yet_mapped_variables else f
                            for m, f in zip(mapped_shape, full_shape)
                        )
                        if isinstance(mapped_sdfg.arrays[prefix + name], dace_data.Array)
                        else "1",
                        dtype=mapped_sdfg.arrays[prefix + name].dtype,
                        strides=mapped_sdfg.arrays[prefix + name].strides,
                        total_size=mapped_sdfg.arrays[prefix + name].total_size,
                        storage=dace.StorageType.Default,
                    )
                else:
                    sdfg.add_scalar(
                        prefix + name,
                        dtype=mapped_sdfg.arrays[prefix + name].dtype,
                        storage=dace.StorageType.Default,
                    )

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = extents[name][i]

                if var in self.not_yet_mapped_variables | {variable.upper()}:
                    # if var == "K":
                    #     subset = library_node.k_range.ranges[0]
                    #     subset_strs.append(
                    #         f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                    #     )
                    # else:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent + upper_extent)}")
                else:
                    subset_strs.append(
                        # f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                        f"0:{-lower_extent + upper_extent + 1}"
                    )
            subset_str = ",".join(subset_strs) if is_array else "0"
            if isinstance(map_scope_delimiter, dace.nodes.MapEntry):
                map_scope_delimiter.add_in_connector("IN_" + name)
                state.add_edge(
                    state.add_read(prefix + name),
                    None,
                    map_scope_delimiter,
                    "IN_" + name,
                    dace.memlet.Memlet.simple(
                        prefix + name, subset_str=subset_str, num_accesses=1
                    ),
                )
            else:
                map_scope_delimiter.add_out_connector("OUT_" + name)
                state.add_edge(
                    map_scope_delimiter,
                    "OUT_" + name,
                    state.add_write(prefix + name),
                    None,
                    dace.memlet.Memlet.simple(
                        prefix + name, subset_str=subset_str, num_accesses=1
                    ),
                )

        if len(names) == 0:
            if isinstance(map_scope_delimiter, dace.nodes.MapEntry):
                state.add_edge(map_scope_delimiter, None, nsdfg_node, None, dace.Memlet())
            else:
                state.add_edge(nsdfg_node, None, map_scope_delimiter, None, dace.Memlet())

    def _map(self, variable):
        limit_var = variable.upper()
        iter_var = variable.lower()

        sdfg = dace.SDFG(self.library_node.label + f"_tmp_{limit_var}_sdfg")
        state = sdfg.add_state(self.library_node.label + f"_tmp_{limit_var}_state")

        symbol_mapping = {}
        for k in self.inner_sdfg.free_symbols:
            symbol_mapping[k] = dace.symbolic.symbol(k, self.inner_sdfg.symbols[k])
            sdfg.symbols[k] = self.inner_sdfg.symbols[k]

        nsdfg_node = state.add_nested_sdfg(
            self.inner_sdfg,
            sdfg,
            set("IN_" + inp for inp in self.library_node.inputs),
            set("OUT_" + outp for outp in self.library_node.outputs),
            symbol_mapping=symbol_mapping,
        )

        ################################### MAP ###################################
        ndrange = {iter_var: f"0:_{limit_var}_loc"}
        map_entry, map_exit = state.add_map(
            self.library_node.label + f"_{variable}_map", ndrange=ndrange
        )

        self._add_edges_map(
            sdfg,
            state,
            nsdfg_node,
            self.library_node.inputs,
            "IN_",
            self.library_node.input_extents,
            map_entry,
            self.library_node.read_accesses,
            variable,
        )
        self._add_edges_map(
            sdfg,
            state,
            nsdfg_node,
            self.library_node.outputs,
            "OUT_",
            self.library_node.output_extents,
            map_exit,
            self.library_node.write_accesses,
            variable,
        )

        from dace.transformation.interstate import InlineSDFG

        sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
        self.inner_sdfg = sdfg
        self.inner_state = state

    def _loop(self, variable):
        limit_var = variable.upper()
        iter_var = variable.lower()

        sdfg = dace.SDFG(self.library_node.label + f"_tmp_{limit_var}_sdfg")
        state = sdfg.add_state(self.library_node.label + f"_tmp_{limit_var}_state")

        symbol_mapping = {}
        for k in self.inner_sdfg.free_symbols:
            symbol_mapping[k] = dace.symbolic.symbol(k, self.inner_sdfg.symbols[k])
            sdfg.symbols[k] = self.inner_sdfg.symbols[k]

        nsdfg_node = state.add_nested_sdfg(
            self.inner_sdfg,
            sdfg,
            set("IN_" + inp for inp in self.library_node.inputs),
            set("OUT_" + outp for outp in self.library_node.outputs),
            symbol_mapping=symbol_mapping,
        )

        ################################### Loop ###################################

        for input in self.library_node.inputs:
            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = self.library_node.input_extents[input][i]

                if var in self.not_yet_mapped_variables:
                    subset_strs.append(
                        f"0:{dace.symbolic.pystr_to_symbolic(f'({self.library_node.ranges[2][1]}) - ({self.library_node.ranges[2][0]})')}+{(-lower_extent+upper_extent)}"
                    )
                elif var == variable:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent+upper_extent)+1}"
                    )
                else:
                    subset_strs.append(f"0:{-lower_extent + upper_extent+1}")
            subset_str = ",".join(subset_strs)

            state.add_edge(
                state.add_read("IN_" + input),
                None,
                nsdfg_node,
                "IN_" + input,
                dace.memlet.Memlet.simple(
                    data="IN_" + input, subset_str=subset_str, num_accesses=1
                ),
            )
            mapped_shape = self.library_node.input_extents[input].shape
            # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"_I_loc+{mapped_shape[0]-1}",
                f"_J_loc+{mapped_shape[1]-1}",
                f"_K_loc+{mapped_shape[2]-1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )
            sdfg.add_array(
                "IN_" + input,
                shape=tuple(
                    m if var not in self.not_yet_mapped_variables | {variable} else f
                    for var, m, f in zip("IJK", mapped_shape, full_shape)
                ),
                dtype=self.inner_sdfg.arrays["IN_" + input].dtype,
                strides=self.inner_sdfg.arrays["IN_" + input].strides,
                total_size=self.inner_sdfg.arrays["IN_" + input].total_size,
                storage=dace.StorageType.Default,
            )

        for output in self.library_node.outputs:
            is_array = isinstance(self.inner_sdfg.arrays["OUT_" + output], dace_data.Array)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = self.library_node.output_extents[output][i]

                if var in self.not_yet_mapped_variables:
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

            state.add_edge(
                nsdfg_node,
                "OUT_" + output,
                state.add_write("OUT_" + output),
                None,
                dace.memlet.Memlet.simple(
                    data="OUT_" + output, subset_str=subset_str, num_accesses=1
                ),
            )
            mapped_shape = self.library_node.output_extents[output].shape
            # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"_I_loc+{mapped_shape[0]-1}",
                f"_J_loc+{mapped_shape[1]-1}",
                f"_K_loc+{mapped_shape[2]-1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )
            if output not in sdfg.arrays:
                if is_array:
                    sdfg.add_array(
                        "OUT_" + output,
                        shape=tuple(
                            m if m in self.not_yet_mapped_variables else f
                            for m, f in zip(mapped_shape, full_shape)
                        )
                        if isinstance(self.inner_sdfg.arrays["OUT_" + output], dace_data.Array)
                        else "1",
                        dtype=self.inner_sdfg.arrays["OUT_" + output].dtype,
                        strides=self.inner_sdfg.arrays["OUT_" + output].strides,
                        total_size=self.inner_sdfg.arrays["OUT_" + output].total_size,
                        storage=dace.StorageType.Default,
                    )
                else:
                    sdfg.add_scalar(
                        "OUT_" + output,
                        dtype=self.inner_sdfg.arrays["OUT_" + output].dtype,
                        storage=dace.StorageType.Default,
                    )

        if self.library_node.iteration_order == gt_ir.IterationOrder.BACKWARD:
            sdfg.add_loop(
                sdfg.add_state(f"_tmp_{limit_var}_init_state"),
                state,
                sdfg.add_state(f"_tmp_{limit_var}_end_state"),
                loop_var=iter_var,
                initialize_expr=f"_{limit_var}_loc-1",
                condition_expr=f"{iter_var}>=0",
                increment_expr=f"{iter_var}-1",
            )
        else:
            sdfg.add_loop(
                sdfg.add_state(f"_tmp_{limit_var}_init_state"),
                state,
                sdfg.add_state(f"_tmp_{limit_var}_end_state"),
                loop_var=iter_var,
                initialize_expr=f"0",
                condition_expr=f"{iter_var}<_{limit_var}_loc",
                increment_expr=f"{iter_var}+1",
            )

        from dace.transformation.interstate import InlineSDFG

        sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
        self.inner_sdfg = sdfg
        self.inner_state = state

    def _expand_tasklet(self):
        """
        sets the inner_sdfg attribute to an sdfg which connects the data acesses per grid point with
        a tasklet that is based on the `code` property (i.e. the tasklet as a nested sdfg.)
        """
        # sdfg = dace.SDFG(self.library_node.label + "_sdfg")
        # self.state = self.sdfg.add_state(self.library_node.label + "_state")

        sdfg = dace.SDFG(self.library_node.label + "_tmp_tasklet_sdfg")
        state = sdfg.add_state(self.library_node.label + "_tmp_tasklet_state")

        node = self.library_node
        tasklet = state.add_tasklet(
            name=node.name + "_tasklet",
            inputs=set(node.read_accesses.keys()),
            outputs=set(node.write_accesses.keys()),
            code=node.code.as_string,
        )
        for edge in self.parent_state.in_edges(node):
            parent_array = self.parent_sdfg.arrays[edge.data.data]
            sdfg.add_array(
                "IN_" + edge.data.data,
                dtype=parent_array.dtype,
                shape=node.input_extents[edge.data.data].shape,
                strides=parent_array.strides,
                total_size=parent_array.total_size,
                storage=dace.StorageType.Default,
            )
        for edge in self.parent_state.out_edges(node):
            parent_array = self.parent_sdfg.arrays[edge.data.data]
            sdfg.add_array(
                "OUT_" + edge.data.data,
                dtype=parent_array.dtype,
                shape=node.output_extents[edge.data.data].shape,
                strides=parent_array.strides,
                total_size=parent_array.total_size,
                storage=dace.StorageType.Default,
            )
        read_accessors = dict()
        write_accessors = dict()

        for name in set(acc.outer_name for acc in node.read_accesses.values()):
            read_accessors[name] = state.add_read("IN_" + name)
        for name in set(acc.outer_name for acc in node.write_accesses.values()):
            write_accessors[name] = state.add_write("OUT_" + name)

        for name, acc in node.read_accesses.items():
            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = ",".join(
                str(o - e)
                for o, e in zip(offset_tuple, node.input_extents[acc.outer_name].lower_indices)
            )
            state.add_edge(
                read_accessors[acc.outer_name],
                None,
                tasklet,
                name,
                dace.memlet.Memlet.simple(
                    "IN_" + acc.outer_name, subset_str=subset_str, num_accesses=1
                ),
            )
        for name, acc in node.write_accesses.items():
            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = ",".join(
                str(o - e)
                for o, e in zip(offset_tuple, node.output_extents[acc.outer_name].lower_indices)
            )
            state.add_edge(
                tasklet,
                name,
                write_accessors[acc.outer_name],
                None,
                dace.memlet.Memlet.simple(
                    "OUT_" + acc.outer_name, subset_str=subset_str, num_accesses=1
                ),
            )
        for k in sdfg.free_symbols:
            sdfg.symbols[k] = self.parent_sdfg.symbols[k]
        self.inner_sdfg = sdfg
        self.inner_state = state

    def _expand(self):
        self._expand_tasklet()

        for variable in reversed(self.library_node.loop_order):
            if (
                variable == "K"
                and self.library_node.iteration_order is not gt_ir.IterationOrder.PARALLEL
            ):
                self._loop(variable=variable)
            else:
                self._map(variable=variable)
        return self.inner_sdfg

    @staticmethod
    def expand(node, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState) -> dace.SDFG:
        expander = StencilExpander(parent_sdfg, parent_state, node)
        return expander._expand()


@dace.library.expansion
class StencilExpandTransformation(dace.library.ExpandTransformation):

    environments = []
    #
    # def _subsets(self, extents):

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg):

        inner_sdfg = StencilExpander.expand(node, parent_sdfg, parent_state)

        symbol_mapping = {}
        for k in inner_sdfg.free_symbols:
            symbol_mapping[k] = k

        symbol_mapping.update(
            _I_loc=f"I+{node.ranges[0][1] - node.ranges[0][0]}",
            _J_loc=f"J+{node.ranges[1][1] - node.ranges[1][0]}",
            _K_loc=f"({node.ranges[2][1]}) - ({node.ranges[2][0]})",
        )

        from dace.transformation.interstate import InlineSDFG

        inner_sdfg.apply_transformations_repeated(InlineSDFG, validate=False)

        from gt4py.backend.dace.sdfg.api import replace_recursive

        for k in "IJK":
            replace_recursive(inner_sdfg, f"_{k}_loc", str(symbol_mapping[f"_{k}_loc"]))

        from gt4py.backend.dace.sdfg.transforms import RemoveTrivialLoop
        from gt4py.backend.dace.sdfg.api import apply_transformations_repeated_recursive
        from dace.transformation.interstate import EndStateElimination, StateAssignElimination
        from dace.transformation.dataflow import MapCollapse

        apply_transformations_repeated_recursive(
            inner_sdfg,
            [RemoveTrivialLoop, EndStateElimination, StateAssignElimination, MapCollapse],
            validate=False,
        )
        inner_sdfg.apply_strict_transformations(validate=False)
        dace.propagate_memlets_sdfg(inner_sdfg)
        res = dace.nodes.NestedSDFG(
            label=node.label,
            sdfg=inner_sdfg,
            inputs=node.in_connectors,
            outputs=node.out_connectors,
            symbol_mapping=symbol_mapping,
        )
        res.sdfg.parent = parent_state
        res.sdfg.parent_sdfg = parent_sdfg
        return res
