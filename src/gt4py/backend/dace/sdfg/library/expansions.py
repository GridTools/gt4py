import dace


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

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):

        tasklet = parent_state.add_tasklet(
            name=node.name + "_tasklet",
            inputs={key: dace.memlet.EmptyMemlet() for key in node.inputs},
            outputs=node.iniputs,
            code=node.code,
        )

        for variable in ForLoopExpandTransformation.iteration_order:
            if variable in "IJ" or node.iteration_order == "parallel":
                pass
            else:
                pass

        return parent_sdfg
