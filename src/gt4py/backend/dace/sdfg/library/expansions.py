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
        input_names,
        output_names,
        input_extents,
        map_entry,
        map_exit,
        variable,
    ):
        mapped_sdfg = nsdfg_node.sdfg

        subsets = {}
        names = set(input_names) | set(output_names)
        extents = dict()
        for name in names:
            # for name, extent in input_extents.items():
            output_extent = (
                gt_ir.Extent([(0, 0), (0, 0), (0, 0)]) if name in output_names else None
            )
            input_extent: gt_ir.Extent = input_extents.get(name, None)
            if output_extent is not None and input_extent is not None:
                extent = input_extent.union(output_extent)
            elif output_extent is not None:
                extent = output_extent
            else:
                assert input_extent is not None
                extent = input_extent
            extents[name] = extent

            subset_strs = []
            for i, var in enumerate("IJK"):

                lower_extent, upper_extent = extent[i]

                if var in self.not_yet_mapped_variables:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent + upper_extent)}")
                elif var == variable:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent + upper_extent) + 1}"
                    )
                else:
                    subset_strs.append(f"0:{-lower_extent + upper_extent +1}")
            subsets[name] = ",".join(subset_strs)

        for name in names:
            is_array = isinstance(mapped_sdfg.arrays[name], dace_data.Array)

            mapped_shape = extents[name].shape
            full_shape = (
                f"_I_loc+{mapped_shape[0] - 1}",
                f"_J_loc+{mapped_shape[1] - 1}",
                f"_K_loc+{mapped_shape[2] - 1}",
            )
            if name not in sdfg.arrays:
                if is_array:
                    sdfg.add_array(
                        name,
                        shape=tuple(
                            f if var in self.not_yet_mapped_variables or var == variable else m
                            for var, m, f in zip("IJK", mapped_shape, full_shape)
                        ),
                        dtype=mapped_sdfg.arrays[name].dtype,
                        strides=mapped_sdfg.arrays[name].strides,
                        total_size=mapped_sdfg.arrays[name].total_size,
                        storage=dace.StorageType.Default,
                    )
                else:
                    sdfg.add_scalar(
                        name,
                        dtype=mapped_sdfg.arrays[name].dtype,
                        storage=dace.StorageType.Default,
                    )
            if name in input_names:
                state.add_memlet_path(
                    state.add_read(name),
                    map_entry,
                    nsdfg_node,
                    dst_conn=name,
                    memlet=dace.memlet.Memlet.simple(
                        data=name, subset_str=subsets[name], num_accesses=1
                    ),
                    propagate=False,
                )
            if name in output_names:
                state.add_memlet_path(
                    nsdfg_node,
                    map_exit,
                    state.add_write(name),
                    src_conn=name,
                    memlet=dace.memlet.Memlet.simple(
                        data=name, subset_str=subsets[name], num_accesses=1
                    ),
                    propagate=False,
                )
        dace.propagate_memlets_sdfg(sdfg)

        if len(input_names) == 0:
            state.add_edge(map_entry, None, nsdfg_node, None, dace.Memlet())
        if len(output_names) == 0:
            state.add_edge(nsdfg_node, None, map_exit, None, dace.Memlet())

    def _map(self, variable):
        limit_var = variable.upper()
        iter_var = variable.lower()

        for i, inner_sdfg in enumerate(self._inner_sdfgs):
            sdfg = dace.SDFG(self.library_node.label + f"_tmp_{limit_var}_sdfg")
            state = sdfg.add_state(self.library_node.label + f"_tmp_{limit_var}_state")
            for var in "IJK":
                sdfg.add_symbol(f"_{var}_loc", dace.dtypes.DTYPE_TO_TYPECLASS[int])

            symbol_mapping = {}
            for k in inner_sdfg.free_symbols:
                symbol_mapping[k] = dace.symbolic.symbol(k, inner_sdfg.symbols[k])
                sdfg.symbols[k] = inner_sdfg.symbols[k]

            nsdfg_node = state.add_nested_sdfg(
                inner_sdfg,
                sdfg,
                set(inp for inp in self.library_node.inputs),
                set(outp for outp in self.library_node.outputs),
                symbol_mapping=symbol_mapping,
            )

            ################################### MAP ###################################
            ndrange = {iter_var: f"0:_{limit_var}_loc"}
            map_entry, map_exit = state.add_map(
                self.library_node.label + f"_{variable}_map", ndrange=ndrange
            )

            inputs = {
                n.data
                for n, _ in inner_sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.AccessNode)
                and n.access != dace.dtypes.AccessType.WriteOnly
            }
            outputs = {
                n.data
                for n, _ in inner_sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.AccessNode)
                and n.access != dace.dtypes.AccessType.ReadOnly
            }
            self._add_edges_map(
                sdfg,
                state,
                nsdfg_node,
                inputs,
                outputs,
                self._input_extents[i],
                map_entry,
                map_exit,
                variable,
            )

            from dace.transformation.interstate import InlineSDFG

            # sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
            self._inner_sdfgs[i] = sdfg

    def _loop(self, variable):
        limit_var = variable.upper()
        iter_var = variable.lower()

        for i, inner_sdfg in enumerate(self._inner_sdfgs):
            sdfg = dace.SDFG(self.library_node.label + f"_tmp_{limit_var}_sdfg")
            state = sdfg.add_state(self.library_node.label + f"_tmp_{limit_var}_state")
            for var in "IJK":
                sdfg.add_symbol(f"_{var}_loc", dace.dtypes.DTYPE_TO_TYPECLASS[int])

            outputs = {
                n.data
                for n, _ in inner_sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.AccessNode)
                and n.access != dace.dtypes.AccessType.ReadOnly
            }
            symbol_mapping = {}
            for k in inner_sdfg.free_symbols:
                symbol_mapping[k] = dace.symbolic.symbol(k, inner_sdfg.symbols[k])
                sdfg.symbols[k] = inner_sdfg.symbols[k]

            nsdfg_node = state.add_nested_sdfg(
                inner_sdfg,
                sdfg,
                set(inp for inp in self.library_node.inputs),
                set(outp for outp in self.library_node.outputs),
                symbol_mapping=symbol_mapping,
            )

            ################################### Loop ###################################
            for input in self.library_node.inputs:
                extent = self._input_extents[i][input]
                if input in outputs:
                    extent = extent.union(gt_ir.Extent([(0, 0), (0, 0), (0, 0)]))
                subset_strs = []
                for var_idx, var in enumerate("IJK"):
                    lower_extent, upper_extent = extent[var_idx]

                    if var == variable:
                        subset_strs.append(
                            f"{var.lower()}:{var.lower()}+{(-lower_extent+upper_extent)+1}"
                        )
                    elif var in self.not_yet_mapped_variables:
                        subset_strs.append(
                            f"0:{dace.symbolic.pystr_to_symbolic(f'_{var}_loc')}+{(-lower_extent + upper_extent)}"
                        )
                    else:
                        subset_strs.append(f"0:{-lower_extent + upper_extent+1}")
                subset_str = ",".join(subset_strs)

                state.add_edge(
                    state.add_read(input),
                    None,
                    nsdfg_node,
                    input,
                    dace.memlet.Memlet.simple(data=input, subset_str=subset_str, num_accesses=1),
                )
                extent = self._input_extents[i][input]
                if input in self.library_node.outputs:
                    extent = extent.union(gt_ir.Extent([(0, 0), (0, 0), (0, 0)]))
                mapped_shape = extent.shape
                # k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
                full_shape = (
                    f"_I_loc+{mapped_shape[0]-1}",
                    f"_J_loc+{mapped_shape[1]-1}",
                    f"_K_loc+{mapped_shape[2]-1}"
                    # f"{k_shape}+({mapped_shape[2]-1})",
                )
                sdfg.add_array(
                    input,
                    shape=tuple(
                        f if var in self.not_yet_mapped_variables or var == variable else m
                        for var, m, f in zip("IJK", mapped_shape, full_shape)
                    ),
                    dtype=inner_sdfg.arrays[input].dtype,
                    strides=inner_sdfg.arrays[input].strides,
                    total_size=inner_sdfg.arrays[input].total_size,
                    storage=dace.StorageType.Default,
                )

            for output in self.library_node.outputs:
                is_array = isinstance(inner_sdfg.arrays[output], dace_data.Array)

                subset_strs = []
                extent = gt_ir.Extent([(0, 0), (0, 0), (0, 0)])
                if output in self._input_extents[i]:
                    extent = extent.union(self._input_extents[i][output])
                for var_idx, var in enumerate("IJK"):
                    lower_extent, upper_extent = extent[var_idx]

                    if var == variable:
                        assert var == "K"
                        subset_strs.append(
                            f"{var.lower()}:{var.lower()}+{(-lower_extent+upper_extent)+1}"
                        )
                    elif var in self.not_yet_mapped_variables:
                        subset_strs.append(
                            f"0:{dace.symbolic.pystr_to_symbolic(f'_{var}_loc')}+{(-lower_extent + upper_extent)}"
                        )
                    else:
                        subset_strs.append(f"0:{-lower_extent + upper_extent+1}")
                subset_str = ",".join(subset_strs) if is_array else "0"

                state.add_edge(
                    nsdfg_node,
                    output,
                    state.add_write(output),
                    None,
                    dace.memlet.Memlet.simple(data=output, subset_str=subset_str, num_accesses=1),
                )
                mapped_shape = (1, 1, 1)
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
                            output,
                            shape=tuple(
                                f if var in self.not_yet_mapped_variables or var == variable else m
                                for var, m, f in zip("IJK", mapped_shape, full_shape)
                            )
                            if isinstance(inner_sdfg.arrays[output], dace_data.Array)
                            else "1",
                            dtype=inner_sdfg.arrays[output].dtype,
                            strides=inner_sdfg.arrays[output].strides,
                            total_size=inner_sdfg.arrays[output].total_size,
                            storage=dace.StorageType.Default,
                        )
                    else:
                        sdfg.add_scalar(
                            output,
                            dtype=inner_sdfg.arrays[output].dtype,
                            storage=dace.StorageType.Default,
                        )

            guard_state = sdfg.add_state(self.library_node.label + f"_guard")
            if self.library_node.iteration_order == gt_ir.IterationOrder.BACKWARD:
                initialize_expr = f"_{limit_var}_loc-1"
                condition_expr = f"{iter_var}>=0"
                increment_expr = f"{iter_var}-1"
            else:
                initialize_expr = f"0"
                condition_expr = f"{iter_var}<_{limit_var}_loc"
                increment_expr = f"{iter_var}+1"
            init_state = sdfg.add_state(self.library_node.label + f"_tmp_{limit_var}_init_state")
            exit_state = sdfg.add_state(self.library_node.label + f"_tmp_{limit_var}_end_state")
            from dace import properties as dace_properties

            sdfg.add_edge(
                init_state,
                guard_state,
                dace.InterstateEdge(assignments={iter_var: initialize_expr}),
            )

            sdfg.add_edge(
                guard_state,
                state,
                dace.InterstateEdge(condition=dace_properties.CodeBlock(condition_expr).code),
            )
            sdfg.add_edge(
                state, guard_state, dace.InterstateEdge(assignments={iter_var: increment_expr})
            )
            sdfg.add_edge(
                guard_state,
                exit_state,
                dace.InterstateEdge(
                    condition=dace_properties.CodeBlock(f"not ({condition_expr})").code
                ),
            )

            from dace.transformation.interstate import InlineSDFG

            # sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
            self._inner_sdfgs[i] = sdfg

    def _make_state_machine(self):
        sdfg = dace.SDFG(self.library_node.label + f"_tmp_state_machine_sdfg")
        state = sdfg.add_state(
            self.library_node.label + f"_tmp_state_machine_start_state", is_start_state=True
        )
        for var in "IJK":
            sdfg.add_symbol(f"_{var}_loc", dace.dtypes.DTYPE_TO_TYPECLASS[int])
        glob_min_k, glob_max_k = self._k_ranges[0][0], self._k_ranges[-1][1]

        input_extents = dict()
        for interval, extents in zip(self._k_ranges, self._input_extents):
            for k, extent in extents.items():
                loc_min_k, loc_max_k = interval
                i_extent, j_extent, (loc_lower, loc_upper) = extent
                k_extent_wrt_global = (
                    int(loc_lower + (loc_min_k - glob_min_k)),
                    int(loc_upper + (loc_max_k - glob_max_k)),
                )
                extent_wrt_global = gt_ir.nodes.Extent(i_extent, j_extent, k_extent_wrt_global)
                if k not in input_extents:
                    input_extents[k] = extent_wrt_global
                input_extents[k] = input_extents[k].union(extent_wrt_global)

        in_out_extents = dict(input_extents)
        for interval, inner_sdfg in zip(self._k_ranges, self._inner_sdfgs):
            output_fields = [
                n.data
                for n, _ in inner_sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.AccessNode)
                and n.access != dace.dtypes.AccessType.ReadOnly
            ]
            for name in output_fields:
                loc_min_k, loc_max_k = interval
                k_extent_wrt_global = (
                    int(loc_min_k - glob_min_k),
                    int(loc_max_k - glob_max_k),
                )
                extent_wrt_global = gt_ir.nodes.Extent((0, 0), (0, 0), k_extent_wrt_global)
                if name not in in_out_extents:
                    in_out_extents[name] = extent_wrt_global
                in_out_extents[name] = in_out_extents[name].union(extent_wrt_global)

        array_dict = dict()
        for inner_sdfg in self._inner_sdfgs:
            for name, datadesc in inner_sdfg.arrays.items():
                array_dict[name] = datadesc
        for k, extent in in_out_extents.items():
            mapped_shape = extent.shape
            full_shape = (
                f"_I_loc+{mapped_shape[0] - 1}",
                f"_J_loc+{mapped_shape[1] - 1}",
                f"_K_loc+{mapped_shape[2] - 1}"
                # f"{k_shape}+({mapped_shape[2]-1})",
            )
            sdfg.add_array(
                k,
                shape=tuple(
                    f if var in self.not_yet_mapped_variables or var == "K" else m
                    for var, m, f in zip("IJK", mapped_shape, full_shape)
                ),
                dtype=array_dict[k].dtype,
                strides=array_dict[k].strides,
            )

        subsets = dict()
        for name, extent in in_out_extents.items():

            subset_strs = []
            for i, var in enumerate("IJK"):

                lower_extent, upper_extent = extent[i]

                if var in self.not_yet_mapped_variables:
                    subset_strs.append(f"{0}:_{var.upper()}_loc+{(-lower_extent + upper_extent)}")
                elif var == "K":
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{(-lower_extent + upper_extent) + 1}"
                    )
                else:
                    subset_strs.append(f"0:{-lower_extent + upper_extent + 1}")
            subsets[name] = ",".join(subset_strs)

        for i, inner_sdfg in enumerate(self._inner_sdfgs):
            old_state = state
            state = sdfg.add_state(self.library_node.label + f"_tmp_state_machine_state_{i}")
            sdfg.add_edge(old_state, state, dace.InterstateEdge())
            symbol_mapping = {}
            for k in inner_sdfg.free_symbols:
                symbol_mapping[k] = dace.symbolic.symbol(k, inner_sdfg.symbols[k])
                sdfg.symbols[k] = inner_sdfg.symbols[k]

            accessors = []
            for s in inner_sdfg.nodes():
                for n in s.nodes():
                    if isinstance(n, dace.nodes.AccessNode):
                        accessors.append(n)

            input_accessors = [
                n for n in accessors if n.access != dace.dtypes.AccessType.WriteOnly
            ]
            output_accessors = [
                n for n in accessors if n.access != dace.dtypes.AccessType.ReadOnly
            ]

            nsdfg_node = state.add_nested_sdfg(
                inner_sdfg,
                sdfg,
                inputs=set(accessor.data for accessor in input_accessors),
                outputs=set(accessor.data for accessor in output_accessors),
                symbol_mapping=symbol_mapping,
            )

            for acc in input_accessors:
                name = acc.data
                inner_extent = self._input_extents[i][name]
                if name in output_accessors:
                    inner_extent = inner_extent.union(gt_ir.Extent([(0, 0), (0, 0), (0, 0)]))
                global_extent = in_out_extents[name]
                i_subset = "{offset_low}: {var} + {offset_high}".format(
                    offset_low=inner_extent[0][0] - global_extent[0][0],
                    var="_I_loc" if "I" in self.not_yet_mapped_variables else 1,
                    offset_high=inner_extent[0][1] - global_extent[0][0],
                )
                j_subset = "{offset_low}: {var} + {offset_high}".format(
                    offset_low=inner_extent[1][0] - global_extent[1][0],
                    var="_J_loc" if "J" in self.not_yet_mapped_variables else 1,
                    offset_high=inner_extent[1][1] - global_extent[1][0],
                )
                k_subset = "{offset_low}: _K_loc + {offset_high}".format(
                    offset_low=inner_extent[2][0] - global_extent[2][0],
                    offset_high=inner_extent[2][1] - global_extent[2][0],
                )
                subset_str = ",".join([i_subset, j_subset, k_subset])
                state.add_edge(
                    state.add_read(name),
                    None,
                    nsdfg_node,
                    name,
                    dace.Memlet.simple(data=name, subset_str=subset_str),
                )
            for acc in output_accessors:
                name = acc.data
                inner_extent = gt_ir.Extent([(0, 0), (0, 0), (0, 0)])
                if name in self._input_extents[i]:
                    inner_extent = inner_extent.union(self._input_extents[i][name])
                global_extent = in_out_extents[name]
                i_subset = "{offset_low}: {var} + {offset_high}".format(
                    offset_low=inner_extent[0][0] - global_extent[0][0],
                    var="_I_loc" if "I" in self.not_yet_mapped_variables else 1,
                    offset_high=inner_extent[0][1] - global_extent[0][0],
                )
                j_subset = "{offset_low}: {var} + {offset_high}".format(
                    offset_low=inner_extent[1][0] - global_extent[1][0],
                    var="_J_loc" if "J" in self.not_yet_mapped_variables else 1,
                    offset_high=inner_extent[1][1] - global_extent[1][0],
                )
                k_subset = "{offset_low}: _K_loc + {offset_high}".format(
                    offset_low=inner_extent[2][0] - global_extent[2][0],
                    offset_high=inner_extent[2][1] - global_extent[2][0],
                )
                subset_str = ",".join([i_subset, j_subset, k_subset])
                state.add_edge(
                    nsdfg_node,
                    name,
                    state.add_write(name),
                    None,
                    dace.Memlet.simple(data=name, subset_str=subset_str),
                )

        self._inner_sdfgs = [sdfg]
        self._k_ranges = [(glob_min_k, glob_max_k)]
        self._input_extents = [input_extents]

    def _expand(self):
        self._inner_sdfgs = [int.sdfg for int in self.library_node.intervals]
        self._input_extents = [int.input_extents for int in self.library_node.intervals]
        self._k_ranges = [int.k_interval for int in self.library_node.intervals]

        for variable in reversed(self.library_node.loop_order):
            if (
                variable == "K"
                and self.library_node.iteration_order is not gt_ir.IterationOrder.PARALLEL
            ):
                self._loop(variable=variable)
            else:
                self._map(variable=variable)
            if variable == "K":
                self._make_state_machine()

            self.not_yet_mapped_variables.add(variable)

        assert len(self._inner_sdfgs) == 1
        assert len(self._k_ranges) == 1

        inner_sdfg = self._inner_sdfgs[0]
        k_range = self._k_ranges[0]

        symbol_mapping = {}
        for k in inner_sdfg.free_symbols:
            symbol_mapping[k] = k

        symbol_mapping.update(
            _I_loc=f"I+{self.library_node.ij_range[0][1] - self.library_node.ij_range[0][0]}",
            _J_loc=f"J+{self.library_node.ij_range[1][1] - self.library_node.ij_range[1][0]}",
            _K_loc=f"({k_range[1]}) - ({k_range[0]})",
        )

        return self._inner_sdfgs[0], symbol_mapping

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

        inner_sdfg, symbol_mapping = StencilExpander.expand(node, parent_sdfg, parent_state)

        from dace.transformation.interstate import InlineSDFG

        # inner_sdfg.apply_transformations_repeated(InlineSDFG, validate=False)

        from gt4py.backend.dace.sdfg.api import replace_recursive

        for k in "IJK":
            replace_recursive(inner_sdfg, f"_{k}_loc", str(symbol_mapping[f"_{k}_loc"]))

        from gt4py.backend.dace.sdfg.transforms import RemoveTrivialLoop
        from gt4py.backend.dace.sdfg.api import apply_transformations_repeated_recursive
        from dace.transformation.interstate import (
            EndStateElimination,
            StateAssignElimination,
            StateFusion,
        )
        from dace.transformation.dataflow import MapCollapse

        apply_transformations_repeated_recursive(
            inner_sdfg,
            [
                StateFusion,
                # RemoveTrivialLoop,
                EndStateElimination,
                StateAssignElimination,
                MapCollapse,
            ],
            validate=False,
        )
        # inner_sdfg.apply_strict_transformations(validate=False)
        in_connectors = {name[len("IN_") :] for name in node.in_connectors.keys()}
        out_connectors = {name[len("OUT_") :] for name in node.out_connectors.keys()}
        res = dace.nodes.NestedSDFG(
            label=node.label,
            sdfg=inner_sdfg,
            inputs=in_connectors,
            outputs=out_connectors,
            symbol_mapping=symbol_mapping,
        )

        for edge in parent_state.in_edges(node):
            edge.dst_conn = edge.dst_conn[len("IN_") :]
        for edge in parent_state.out_edges(node):
            edge.src_conn = edge.src_conn[len("OUT_") :]
        for edge in parent_state.in_edges(node):
            for other_edge in parent_state.out_edges(node):
                if other_edge.data.data == edge.data.data:
                    edge.data.subset = dace.memlet.subsets.union(
                        edge.data.subset, other_edge.data.subset
                    )

        for edge in parent_state.out_edges(node):
            for other_edge in parent_state.in_edges(node):
                if other_edge.data.data == edge.data.data:
                    edge.data.subset = dace.memlet.subsets.union(
                        edge.data.subset, other_edge.data.subset
                    )

        res.sdfg.parent = parent_state
        res.sdfg.parent_sdfg = parent_sdfg
        return res
