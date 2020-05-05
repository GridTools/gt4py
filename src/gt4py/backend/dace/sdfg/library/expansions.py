import dace
import dace.subsets

from gt4py import ir as gt_ir


@dace.library.expansion
class ForLoopExpandTransformation(dace.library.ExpandTransformation):

    environments = []

    iteration_order = "IJK"

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

        nsdfg_node = tmp_state.add_nested_sdfg(
            mapped_sdfg, tmp_sdfg, library_node.in_connectors, library_node.out_connectors
        )

        ndrange = {iter_var: str(library_node.k_range) if limit_var == "K" else f"0:{limit_var}"}
        map_entry, map_exit = tmp_state.add_map(
            library_node.label + f"_{variable}_map", ndrange=ndrange
        )
        for input in library_node.in_connectors:
            assert input in nsdfg_node.in_connectors
            output = "OUT_" + input[3:]

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.input_extents[input[3:]][i]

                if var in not_yet_mapped_vars:
                    if var == "K":
                        subset = library_node.k_range.ranges[0]
                        subset_strs.append(
                            f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                        )
                    else:
                        subset_strs.append(f"{0}:{var.upper()}+{(-lower_extent+upper_extent+1)}")
                else:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs)

            map_entry.add_out_connector(output)
            tmp_state.add_edge(
                map_entry,
                output,
                nsdfg_node,
                input,
                dace.memlet.Memlet.simple(data=input[3:], subset_str=subset_str),
            )
            mapped_shape = library_node.input_extents[input[3:]].shape
            k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"I+{mapped_shape[0]-1}",
                f"J+{mapped_shape[1]-1}",
                f"{k_shape}+({mapped_shape[2]-1})",
            )
            tmp_sdfg.add_array(
                input[3:],
                shape=tuple(
                    m if m in not_yet_mapped_vars else f for m, f in zip(mapped_shape, full_shape)
                ),
                dtype=mapped_sdfg.arrays[input[3:]].dtype,
            )
            map_entry.add_in_connector(input)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.input_extents[input[3:]][i]

                if var in not_yet_mapped_vars | {variable.upper()}:
                    if var == "K":
                        subset = library_node.k_range.ranges[0]
                        subset_strs.append(
                            f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                        )
                    else:
                        subset_strs.append(f"{0}:{var.upper()}+{(-lower_extent+upper_extent+1)}")
                else:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs)

            tmp_state.add_edge(
                tmp_state.add_read(input[3:]),
                None,
                map_entry,
                input,
                dace.memlet.Memlet.simple(input[3:], subset_str=subset_str),
            )

        for output in library_node.out_connectors:
            assert output in nsdfg_node.out_connectors
            input = "IN_" + output[4:]

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.output_extents[output[4:]][i]

                if var in not_yet_mapped_vars:
                    if var == "K":
                        subset = library_node.k_range.ranges[0]
                        subset_strs.append(
                            f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                        )
                    else:
                        subset_strs.append(f"{0}:{var.upper()}+{(-lower_extent+upper_extent+1)}")
                else:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs)

            map_exit.add_in_connector(input)
            tmp_state.add_edge(
                nsdfg_node,
                output,
                map_exit,
                input,
                dace.memlet.Memlet.simple(data=output[4:], subset_str=subset_str),
            )
            mapped_shape = library_node.output_extents[output[4:]].shape
            k_shape = library_node.k_range.ranges[0][1] - library_node.k_range.ranges[0][0] + 1
            full_shape = (
                f"I+{mapped_shape[0]-1}",
                f"J+{mapped_shape[1]-1}",
                f"{k_shape}+({mapped_shape[2]-1})",
            )
            tmp_sdfg.add_array(
                output[4:],
                shape=tuple(
                    m if m in not_yet_mapped_vars else f for m, f in zip(mapped_shape, full_shape)
                ),
                dtype=mapped_sdfg.arrays[output[4:]].dtype,
            )
            map_exit.add_out_connector(output)

            subset_strs = []
            for i, var in enumerate("IJK"):
                lower_extent, upper_extent = library_node.output_extents[input[3:]][i]

                if var in not_yet_mapped_vars | {variable.upper()}:
                    if var == "K":
                        subset = library_node.k_range.ranges[0]
                        subset_strs.append(
                            f"{subset[0]-lower_extent}:{subset[1]+(-lower_extent+upper_extent+1)}"
                        )
                    else:
                        subset_strs.append(f"{0}:{var.upper()}+{(-lower_extent+upper_extent+1)}")
                else:
                    subset_strs.append(
                        f"{var.lower()}:{var.lower()}+{-lower_extent+upper_extent+1}"
                    )
            subset_str = ",".join(subset_strs)

            tmp_state.add_edge(
                map_exit,
                output,
                tmp_state.add_write(output[4:]),
                None,
                dace.memlet.Memlet.simple(output[4:], subset_str=subset_str),
            )

        tmp_sdfg.save("tmp__map.sdfg")
        print(variable)
        tmp_sdfg.validate()
        # from dace.transformation.dataflow import MergeArrays
        from dace.transformation.interstate import InlineSDFG

        tmp_sdfg.apply_transformations_repeated([InlineSDFG], validate=False)
        return tmp_sdfg

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        tasklet_in_connectors = set(node.read_accesses.keys())
        tasklet_out_connectors = set(node.write_accesses.keys())

        tmp_sdfg = dace.SDFG(node.label + "_tmp_tasklet_sdfg")
        tmp_state = tmp_sdfg.add_state(node.label + "_tmp_tasklet_state")
        tasklet = tmp_state.add_tasklet(
            name=node.name + "_tasklet",
            inputs=set(node.read_accesses.keys()),
            outputs=set(node.write_accesses.keys()),
            code=node.code.as_string,
        )

        for name, acc in node.read_accesses.items():
            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = ",".join(
                str(o - e)
                for o, e in zip(offset_tuple, node.input_extents[acc.outer_name].lower_indices)
            )
            tmp_state.add_edge(
                tmp_state.add_read(acc.outer_name),
                None,
                tasklet,
                name,
                dace.memlet.Memlet.simple(acc.outer_name, subset_str=subset_str),
            )
        for name, acc in node.write_accesses.items():
            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = ",".join(
                str(o - e)
                for o, e in zip(offset_tuple, node.output_extents[acc.outer_name].lower_indices)
            )
            tmp_state.add_edge(
                tasklet,
                name,
                tmp_state.add_write(acc.outer_name),
                None,
                dace.memlet.Memlet.simple(acc.outer_name, subset_str=subset_str),
            )

        for edge in parent_state.in_edges(node):
            parent_array = parent_sdfg.arrays[edge.data.data]
            tmp_sdfg.add_array(
                edge.data.data,
                dtype=parent_array.dtype,
                shape=node.input_extents[edge.data.data].shape,
            )
        for edge in parent_state.out_edges(node):
            parent_array = parent_sdfg.arrays[edge.data.data]
            tmp_sdfg.add_array(
                edge.data.data,
                dtype=parent_array.dtype,
                shape=node.output_extents[edge.data.data].shape,
            )

        tmp_sdfg.validate()
        not_yet_mapped_vars = set()
        for variable in reversed(cls.iteration_order):
            if variable == "K" and node.iteration_order is not gt_ir.IterationOrder.PARALLEL:
                if node.iteration_order is gt_ir.IterationOrder.FORWARD:
                    pass
                if node.iteration_order is gt_ir.IterationOrder.BACKWARD:
                    pass
            else:
                tmp_sdfg = cls._map(
                    node, tmp_sdfg, not_yet_mapped_vars=not_yet_mapped_vars, variable=variable
                )
            not_yet_mapped_vars.add(variable)

        return tmp_sdfg
