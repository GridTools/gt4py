# GridTools Compiler Toolchain (GTC) - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GTC project and the GridTools framework.
# GTC is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import dace

import eve
import gtc.dace.utils
from eve.iterators import iter_tree
from gtc import common
from gtc import daceir as dcir
from gtc import oir
from gtc.dace.utils import union_node_access_infos, union_node_grid_subsets


def set_grid_subset_idx(axis, grid_subset, idx):
    if isinstance(grid_subset.intervals[axis], dcir.IndexWithExtent):
        extent = grid_subset.intervals[axis].extent
    else:
        extent = [0, 0]
    return grid_subset.set_interval(
        axis,
        dcir.IndexWithExtent(
            axis=axis,
            value=idx,
            extent=extent,
        ),
    )


def set_idx(axis, memlets: List[dcir.Memlet], idx):
    res = list()
    for mem in memlets:
        grid_subset = set_grid_subset_idx(axis, mem.access_info.grid_subset, idx)
        res.append(
            dcir.Memlet(
                field=mem.field,
                access_info=dcir.FieldAccessInfo(
                    grid_subset=grid_subset,
                    global_grid_subset=mem.access_info.global_grid_subset,
                    dynamic_access=mem.access_info.dynamic_access,
                    variable_offset_axes=mem.access_info.variable_offset_axes,  # should actually never be True here
                ),
                connector=mem.connector,
                is_read=mem.is_read,
                is_write=mem.is_write,
                other_grid_subset=mem.other_grid_subset,
            )
        )
    return res


class FieldAccessRenamer(eve.NodeMutator):
    def apply(self, node, *, local_name_map):
        return self.visit(node, local_name_map=local_name_map)

    def _rename_memlets(self, memlets, *, local_name_map):
        return [
            dcir.Memlet(
                field=local_name_map[mem.field] if mem.field in local_name_map else mem.field,
                access_info=mem.access_info,
                connector=local_name_map[mem.connector]
                if mem.connector is not None and mem.connector in local_name_map
                else mem.connector,
                is_read=mem.is_read,
                is_write=mem.is_write,
                other_grid_subset=mem.other_grid_subset,
            )
            for mem in memlets
        ]

    def visit_CopyState(self, node: dcir.CopyState, *, local_name_map):
        name_map = {local_name_map[k]: local_name_map[v] for k, v in node.name_map.items()}
        return dcir.CopyState(
            memlets=self._rename_memlets(node.memlets, local_name_map=local_name_map),
            name_map=name_map,
        )

    def visit_DomainMap(self, node: dcir.DomainMap, *, local_name_map):
        computations = self.visit(node.computations, local_name_map=local_name_map)
        return dcir.DomainMap(
            index_ranges=node.index_ranges,
            computations=computations,
            schedule=node.schedule,
            grid_subset=node.grid_subset,
            read_memlets=self._rename_memlets(node.read_memlets, local_name_map=local_name_map),
            write_memlets=self._rename_memlets(node.write_memlets, local_name_map=local_name_map),
        )

    def visit_DomainLoop(self, node: dcir.DomainLoop, *, local_name_map):
        return dcir.DomainLoop(
            axis=node.axis,
            index_range=node.index_range,
            loop_states=self.visit(node.loop_states, local_name_map=local_name_map),
            grid_subset=node.grid_subset,
            read_memlets=self._rename_memlets(node.read_memlets, local_name_map=local_name_map),
            write_memlets=self._rename_memlets(node.write_memlets, local_name_map=local_name_map),
        )

    def visit_NestedSDFG(self, node: dcir.NestedSDFG, *, local_name_map: Dict[str, str]):
        name_map = dict(node.name_map)
        for old_array_name in node.name_map.keys():
            if old_array_name in local_name_map:
                new_array_name = local_name_map[old_array_name]
                name_map[new_array_name] = name_map[old_array_name]
                del name_map[old_array_name]
        return dcir.NestedSDFG(
            label=node.label,
            field_decls=node.field_decls,  # don't rename, this is inside
            read_memlets=self._rename_memlets(node.read_memlets, local_name_map=local_name_map),
            write_memlets=self._rename_memlets(node.write_memlets, local_name_map=local_name_map),
            symbol_decls=node.symbol_decls,
            states=node.states,
            name_map=name_map,
        )

    def visit_Tasklet(self, node: dcir.Tasklet, *, local_name_map):
        name_map = dict(node.name_map)
        for old_array_name in node.name_map.keys():
            if old_array_name in local_name_map:
                new_array_name = local_name_map[old_array_name]
                name_map[new_array_name] = name_map[old_array_name]
                del name_map[old_array_name]
        return dcir.Tasklet(
            decls=node.decls,
            read_memlets=self._rename_memlets(node.read_memlets, local_name_map=local_name_map),
            write_memlets=self._rename_memlets(node.write_memlets, local_name_map=local_name_map),
            name_map=name_map,
            stmts=node.stmts,
        )


rename_field_accesses = FieldAccessRenamer().apply


class FieldDeclPropagator(eve.NodeMutator):
    def apply(self, node, *, decl_map):
        return self.visit(node, decl_map=decl_map)

    def visit_NestedSDFG(
        self,
        node: dcir.NestedSDFG,
        *,
        decl_map: Dict[str, Tuple[Any, dace.StorageType]],
    ):
        field_decls = {decl.name: decl for decl in node.field_decls}
        for name, (strides, storage) in decl_map.items():
            if name in node.name_map:
                orig_field_decl = field_decls[node.name_map[name]]
                field_decls[node.name_map[name]] = dcir.FieldDecl(
                    name=orig_field_decl.name,
                    dtype=orig_field_decl.dtype,
                    strides=strides,
                    data_dims=orig_field_decl.data_dims,
                    access_info=orig_field_decl.access_info,
                    storage=storage,
                )
        states = self.visit(
            node.states,
            decl_map={node.name_map[k]: v for k, v in decl_map.items() if k in node.name_map},
        )
        return dcir.NestedSDFG(
            label=node.label,
            field_decls=[decl for decl in field_decls.values()],  # don't rename, this is inside
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            symbol_decls=node.symbol_decls,
            states=states,
            name_map=node.name_map,
        )


propagate_field_decls = FieldDeclPropagator().apply


class MakeLocalCaches(eve.NodeTranslator):
    def _write_before_read_fields(self, axis, node):
        for state_machine in iter_tree(node).if_isinstance(dcir.NestedSDFG):
            if state_machine is node:
                continue
            res = self._write_before_read_fields(axis, state_machine)
            name_map = {v: k for k, v in state_machine.name_map.items()}
            res = {name_map[k] for k in res}
            return res

        tasklets = iter_tree(node).if_isinstance(dcir.Tasklet).to_list()

        # guaranteed/maybe due to masks
        guaranteed_write_first_fields = set()
        maybe_read_first_fields = set()

        def _iterate_access_order(node, is_write=False, is_masked=None):

            if isinstance(node, list):
                for n in node:
                    yield from _iterate_access_order(n)
            elif isinstance(node, dcir.Tasklet):
                yield from _iterate_access_order(node.stmts)
            elif isinstance(node, dcir.MaskStmt):
                yield from _iterate_access_order(node.mask, is_write=False, is_masked=is_masked)
                yield from _iterate_access_order(node.body, is_masked=True)
            elif isinstance(node, dcir.HorizontalRestriction):
                yield from _iterate_access_order(node.body, is_masked=True)
            elif isinstance(node, dcir.AssignStmt):
                yield from _iterate_access_order(node.right, is_write=False, is_masked=is_masked)
                yield from _iterate_access_order(node.left, is_write=True, is_masked=is_masked)
            else:
                for node in iter_tree(node).if_isinstance(dcir.IndexAccess, dcir.ScalarAccess):
                    yield node, is_write, is_masked

        connector_to_field = {
            mem.connector: mem.field
            for tasklet in tasklets
            for mem in tasklet.read_memlets + tasklet.write_memlets
        }
        for acc, is_write, is_masked in _iterate_access_order(tasklets):
            if (field_name := connector_to_field.get(acc.name)) is None:
                continue
            if is_write and not is_masked:
                if field_name not in maybe_read_first_fields:
                    guaranteed_write_first_fields.add(field_name)
            elif not is_write and field_name not in guaranteed_write_first_fields:
                maybe_read_first_fields.add(field_name)

        tasklet_name_map = {k: v for tasklet in tasklets for k, v in tasklet.name_map.items()}
        name_map = {v: k for k, v in tasklet_name_map.items()}
        res = {name_map[k] for k in guaranteed_write_first_fields}
        return res

    def _subsets_diff(self, axis, left_subsets, right_subsets):
        res = dict(left_subsets)
        for name, right_subset in right_subsets.items():
            if name not in left_subsets:
                continue
            left_subset = left_subsets[name]
            if right_subset is None:
                continue
            if left_subset is None:
                res[name] = None
                continue
            assert isinstance(left_subset.intervals[axis], dcir.IndexWithExtent) and isinstance(
                left_subset.intervals[axis], dcir.IndexWithExtent
            )
            left_idx, right_idx = left_subset.intervals[axis], right_subset.intervals[axis]

            if left_idx.extent == right_idx.extent:
                # subtract entire subset=empty
                res[name] = None
            elif (
                left_idx.extent[0] >= right_idx.extent[0]
                and left_idx.extent[1] <= right_idx.extent[1]
            ):
                # subtract more than entire subset=empty
                res[name] = None
            elif (
                left_idx.extent[0] < right_idx.extent[0]
                and left_idx.extent[1] > right_idx.extent[1]
            ):
                # result not contiguous: keep original subset as bounding box
                res[name] = left_subset
            else:
                # partial overlaps:
                if left_idx.extent[0] < right_idx.extent[0]:
                    extent = left_idx.extent[0], right_idx.extent[0]
                else:
                    extent = right_idx.extent[1], left_idx.extent[1]
                res[name] = left_subset.set_interval(
                    axis, dcir.IndexWithExtent(axis=axis, value=left_idx.value, extent=extent)
                )
        for name, subset in left_subsets.items():
            res.setdefault(name, subset)
        return res

    def _get_cache_read_subset(self, axis, loop_states, local_name_map):
        read_accesses, write_accesses, _ = union_node_access_infos(list(loop_states))
        read_subsets = {
            name: info.grid_subset for name, info in read_accesses.items() if name in local_name_map
        }
        write_subsets = {
            name: info.grid_subset
            for name, info in write_accesses.items()
            if name in local_name_map
        }
        write_before_read_fields = self._write_before_read_fields(axis, loop_states)
        write_before_read_subsets = {
            k: v for k, v in write_subsets.items() if k in write_before_read_fields
        }
        required_cache_subsets = self._subsets_diff(axis, read_subsets, write_before_read_subsets)
        return {k: v for k, v in required_cache_subsets.items() if k in local_name_map}

    def _get_fill_subsets(self, axis, loop_states, stride, local_name_map):
        read_accesses, write_accesses, _ = union_node_access_infos(list(loop_states))
        read_subsets = {name: info.grid_subset for name, info in read_accesses.items()}
        write_before_read_fields = self._write_before_read_fields(axis, loop_states)
        if stride > 0:
            fill_fields = set(
                name
                for name in write_accesses.keys()
                if name in read_accesses
                and axis in read_accesses[name].grid_subset.intervals
                and read_accesses[name].grid_subset.intervals[axis].extent[1] > 0
            ) | set(
                name
                for name in write_accesses.keys()
                if name in read_accesses
                and axis in read_accesses[name].grid_subset.intervals
                and read_accesses[name].grid_subset.intervals[axis].extent[1] == 0
                and name not in write_before_read_fields
            )

        else:
            fill_fields = set(
                name
                for name in write_accesses.keys()
                if name in read_accesses
                and axis in read_accesses[name].grid_subset.intervals
                and read_accesses[name].grid_subset.intervals[axis].extent[0] < 0
            ) | set(
                name
                for name in write_accesses.keys()
                if name in read_accesses
                and axis in read_accesses[name].grid_subset.intervals
                and read_accesses[name].grid_subset.intervals[axis].extent[0] == 0
                and name not in write_before_read_fields
            )
        fill_fields |= {
            name
            for name in set(read_accesses.keys()) - set(write_accesses.keys())
            if name in local_name_map
        }

        fill_subsets = dict()
        for name in fill_fields:
            idx = read_subsets[name].intervals[axis]
            assert isinstance(idx, dcir.IndexWithExtent)
            if stride > 0:
                extent = idx.extent[1], idx.extent[1]
            else:
                extent = idx.extent[0], idx.extent[0]
            fill_subsets[name] = read_subsets[name].set_interval(
                axis, dcir.IndexWithExtent(axis=axis, value=idx.value, extent=extent)
            )
        return {k: v for k, v in fill_subsets.items() if k in local_name_map}

    def _shift_subsets(self, axis, subsets, offset):
        res = dict()
        for name, subset in subsets.items():
            res[name] = subset.set_interval(axis, subset.intervals[axis].shifted(offset))
        return res

    def _set_other_subset(self, memlets, subsets):
        res = list()
        memlet_dict = {mem.field: mem for mem in memlets}
        for name, subset in subsets.items():
            if name in memlet_dict:
                memlet = memlet_dict[name]
                res.append(
                    dcir.Memlet(
                        field=memlet.field,
                        access_info=memlet.access_info,
                        connector=memlet.connector,
                        is_read=memlet.is_read,
                        is_write=memlet.is_write,
                        other_grid_subset=subset,
                    )
                )
        return res

    def _set_access_info_subset(self, memlets, subsets):
        res = list()
        memlet_dict = {mem.field: mem for mem in memlets}
        for name, subset in subsets.items():
            if name in memlet_dict:
                memlet = memlet_dict[name]
                res.append(
                    dcir.Memlet(
                        field=memlet.field,
                        access_info=dcir.FieldAccessInfo(
                            grid_subset=subset,
                            global_grid_subset=memlet.access_info.global_grid_subset,
                            dynamic_access=memlet.access_info.dynamic_access,
                            variable_offset_axes=memlet.access_info.variable_offset_axes,
                        ),
                        connector=memlet.connector,
                        is_read=memlet.is_read,
                        is_write=memlet.is_write,
                        other_grid_subset=memlet.other_grid_subset,
                    )
                )
        return res

    def _make_localcache_states(
        self,
        loop: dcir.DomainLoop,
        loop_states,
        *,
        context_read_accesses,
        context_write_accesses,
        context_field_accesses,
        local_name_map,
        cache_subsets,
    ):

        # also, init

        # grid_subset = union_node_grid_subsets(list(loop_states))
        # read_accesses, write_accesses, field_accesses = union_node_access_infos(list(loop_states))
        read_memlets = gtc.dace.utils.union_memlets(
            *(
                mem
                for st in gtc.dace.utils.collect_toplevel_computation_nodes(loop_states)
                for mem in st.read_memlets
            )
        )
        write_memlets = gtc.dace.utils.union_memlets(
            *(
                mem
                for st in gtc.dace.utils.collect_toplevel_computation_nodes(loop_states)
                for mem in st.write_memlets
            )
        )

        axis = loop.axis

        cache_read_subsets = self._get_cache_read_subset(axis, loop_states, local_name_map)
        fill_subsets = self._get_fill_subsets(
            axis, loop_states, loop.index_range.stride, local_name_map
        )
        flush_subsets = {
            mem.field: mem.access_info.grid_subset
            for mem in write_memlets
            if mem.field in local_name_map
        }
        init_subsets = self._subsets_diff(
            axis, self._subsets_diff(axis, cache_read_subsets, cache_subsets), fill_subsets
        )

        shift_subsets = self._subsets_diff(axis, cache_read_subsets, fill_subsets)
        cache_subsets.clear()
        cache_subsets.update(
            {k: v for k, v in shift_subsets.items() if v is not None},
        )

        def _remove_connector(memlets):
            return [
                dcir.Memlet(
                    field=mem.field,
                    access_info=mem.access_info,
                    connector=None,
                    is_read=mem.is_read,
                    is_write=mem.is_write,
                    other_grid_subset=mem.other_grid_subset,
                )
                for mem in memlets
            ]

        def _set_readonly(memlets):
            return [
                dcir.Memlet(
                    field=mem.field,
                    access_info=mem.access_info,
                    connector=mem.connector,
                    is_read=True,
                    is_write=False,
                    other_grid_subset=mem.other_grid_subset,
                )
                for mem in memlets
            ]

        def _set_local_names(memlets):
            return [
                dcir.Memlet(
                    field=local_name_map[mem.field],
                    access_info=mem.access_info,
                    connector=mem.connector,
                    is_read=mem.is_read,
                    is_write=mem.is_write,
                    other_grid_subset=mem.other_grid_subset,
                )
                for mem in memlets
            ]

        def _shift_src_subset(memlets, offset):
            res = list()
            for memlet in memlets:
                grid_subset = memlet.access_info.grid_subset.set_interval(
                    axis, memlet.access_info.grid_subset.intervals[axis].shifted(offset)
                )

                res.append(
                    dcir.Memlet(
                        field=memlet.field,
                        access_info=dcir.FieldAccessInfo(
                            grid_subset=grid_subset,
                            global_grid_subset=memlet.access_info.global_grid_subset,
                            dynamic_access=memlet.access_info.dynamic_access,
                            variable_offset_axes=memlet.access_info.variable_offset_axes,  # should actually never be True here
                        ),
                        connector=memlet.connector,
                        is_read=memlet.is_read,
                        is_write=memlet.is_write,
                        other_grid_subset=memlet.other_grid_subset,
                    )
                )
            return res

        init_memlets = self._set_other_subset(
            self._set_access_info_subset(
                [
                    mem
                    for mem in gtc.dace.utils.union_memlets(*read_memlets, *write_memlets)
                    if mem.field in init_subsets and init_subsets[mem.field] is not None
                ],
                init_subsets,
            ),
            init_subsets,
        )
        if loop.index_range.stride > 0:
            init_idx = loop.index_range.interval.start
        else:
            int_end = loop.index_range.interval.end
            init_idx = dcir.AxisBound(axis=axis, level=int_end.level, offset=int_end.offset - 1)
        init_state = dcir.CopyState(
            memlets=_remove_connector(_set_readonly(set_idx(axis, init_memlets, init_idx))),
            name_map=local_name_map,
        )

        fill_accesses = self._set_other_subset(
            self._set_access_info_subset(
                [
                    mem
                    for mem in gtc.dace.utils.union_memlets(*read_memlets, *write_memlets)
                    if mem.field in fill_subsets and fill_subsets[mem.field] is not None
                ],
                fill_subsets,
            ),
            fill_subsets,
        )
        fill_state = dcir.CopyState(
            memlets=_remove_connector(_set_readonly(fill_accesses)),
            name_map=local_name_map,
        )
        loop_states = rename_field_accesses(loop_states, local_name_map=local_name_map)

        flush_accesses = self._set_access_info_subset(
            [
                mem
                for mem in gtc.dace.utils.union_memlets(*write_memlets)
                if mem.field in flush_subsets and flush_subsets[mem.field] is not None
            ],
            flush_subsets,
        )
        flush_state = dcir.CopyState(
            memlets=_remove_connector(_set_readonly(_set_local_names(flush_accesses))),
            name_map={v: k for k, v in local_name_map.items()},
        )

        shift_accesses = self._set_other_subset(
            self._set_access_info_subset(
                [
                    mem
                    for mem in gtc.dace.utils.union_memlets(*read_memlets, *write_memlets)
                    if mem.field in shift_subsets and shift_subsets[mem.field] is not None
                ],
                shift_subsets,
            ),
            shift_subsets,
        )

        shift_state = dcir.CopyState(
            memlets=_remove_connector(
                _set_readonly(
                    _shift_src_subset(_set_local_names(shift_accesses), loop.index_range.stride)
                )
            ),
            name_map={v: v for v in local_name_map.values()},
        )
        return init_state, (fill_state, *loop_states, flush_state, shift_state)

    def _process_state_sequence(self, states, *, fields):

        loops_and_states = []
        for loop in states:
            if isinstance(loop, dcir.DomainLoop):
                loops_and_states.append((loop, loop.loop_states))

        if not loops_and_states:
            return states, {}

        loops = [ls[0] for ls in loops_and_states]
        loop_states = [ls[1] for ls in loops_and_states]

        read_accesses, write_accesses, field_accesses = union_node_access_infos(
            [s for ls in loop_states for s in ls]
        )

        assert len(set(loop.axis for loop in loops)) == 1
        axis = loops[0].axis

        cache_fields = set()
        for name in read_accesses.keys():
            if axis not in field_accesses[name].grid_subset.intervals:
                continue
            if axis in field_accesses[name].grid_subset.intervals:
                access = field_accesses[name]
                if access is not None:
                    interval = access.grid_subset.intervals[axis]
                else:
                    interval = None
                dynamic_write = (
                    write_accesses[name].dynamic_access if name in write_accesses else False
                )
                if (
                    name in fields
                    and isinstance(interval, dcir.IndexWithExtent)
                    # and interval.size > 1
                    and not dynamic_write
                ):
                    cache_fields.add(name)

        if not cache_fields:
            return states, {}

        local_name_map = {k: f"__local_{k}" for k in cache_fields}

        res_states = []
        last_loop_node = None
        cache_subsets = dict()
        for loop_node, loop_state_nodes in loops_and_states:
            if last_loop_node is not None:
                if (
                    last_loop_node.index_range.interval.end == loop_node.index_range.interval.start
                    and last_loop_node.index_range.stride == loop_node.index_range.stride
                ):
                    cache_subsets = cache_subsets
                else:
                    # TODO avoid re-init in non-contiguous cases
                    cache_subsets = dict()

            init_state, (
                fill_state,
                *loop_states,
                flush_state,
                shift_state,
            ) = self._make_localcache_states(
                loop_node,
                loop_state_nodes,
                context_read_accesses=read_accesses,
                context_write_accesses=write_accesses,
                context_field_accesses=field_accesses,
                local_name_map=local_name_map,
                cache_subsets=cache_subsets,
            )

            res_states.extend(
                [
                    init_state,
                    dcir.DomainLoop(
                        loop_states=[fill_state, *loop_states, flush_state, shift_state],
                        axis=loop_node.axis,
                        index_range=loop_node.index_range,
                        grid_subset=loop_node.grid_subset,
                        read_memlets=loop_node.read_memlets,
                        write_memlets=loop_node.write_memlets,
                    ),
                ]
            )
            last_loop_node = loop_node
        return res_states, local_name_map

    def visit_DomainLoop(self, node: dcir.DomainLoop, *, ctx_name_map, axis, fields, **kwargs):
        loop_states = node.loop_states
        # first, recurse
        if any(isinstance(n, dcir.DomainLoop) for n in loop_states):
            start_idx = [i for i, n in enumerate(loop_states) if isinstance(n, dcir.DomainLoop)][0]
            end_idx = [
                i
                for i, n in enumerate(loop_states)
                if i > start_idx and not isinstance(n, dcir.DomainLoop)
            ]
            end_idx = end_idx[0] if end_idx else None
            loop_nodes = [n for n in loop_states if isinstance(n, dcir.DomainLoop)]
            if loop_nodes[0].axis == axis:
                domain_loop_nodes, local_name_map = self._process_state_sequence(
                    loop_nodes, fields=fields
                )
                res_loop_states = [*loop_states[:start_idx], *domain_loop_nodes]
                if end_idx:
                    res_loop_states += loop_states[end_idx:]
                loop_states = res_loop_states
                ctx_name_map.update(local_name_map)

        inner_name_map = dict()
        loop_states = self.generic_visit(
            loop_states, fields=fields, ctx_name_map=inner_name_map, axis=axis, **kwargs
        )
        ctx_name_map.update(inner_name_map)

        read_memlets = gtc.dace.utils.union_memlets(
            *(
                mem
                for st in gtc.dace.utils.collect_toplevel_computation_nodes(loop_states)
                for mem in st.read_memlets
            )
        )
        write_memlets = gtc.dace.utils.union_memlets(
            *(
                mem
                for st in gtc.dace.utils.collect_toplevel_computation_nodes(loop_states)
                for mem in st.write_memlets
            )
        )
        return dcir.DomainLoop(
            grid_subset=node.grid_subset,
            read_memlets=read_memlets,
            write_memlets=write_memlets,
            axis=node.axis,
            index_range=node.index_range,
            loop_states=loop_states,
        )

    def _collect_access_infos_from_memlets(self, states, old_decls):
        from gtc.dace.utils import union_node_grid_subsets

        name_and_subsets = (
            [
                (mem.field, mem.access_info.grid_subset)
                for loop in states
                if isinstance(loop, dcir.DomainLoop)
                for n in gtc.dace.utils.collect_toplevel_computation_nodes(loop.loop_states)
                for l in (n.read_memlets, n.write_memlets)
                for mem in l
            ]
            + [
                (mem.field, mem.access_info.grid_subset)
                for n in gtc.dace.utils.collect_toplevel_copy_states(states)
                for mem in n.memlets
            ]
            + [
                (n.name_map[mem.field], mem.other_grid_subset)
                for n in gtc.dace.utils.collect_toplevel_copy_states(states)
                for mem in n.memlets
            ]
        )
        subsets_dict = dict()
        for name, subset in name_and_subsets:
            subsets_dict[name] = subset.union(subsets_dict.get(name, subset))
        return subsets_dict
        # field_decls = {decl.name: decl for decl in old_decls}
        # field_accesses = {
        #     mem.field: mem.access_info
        #     if mem.field.startswith("__local_")
        #     else field_decls[mem.field].access_info
        #     for mem in
        # }
        # return field_accesses

    def visit_NestedSDFG(self, node: dcir.NestedSDFG, *, axis, fields, storage, **kwargs):
        states = node.states
        local_name_map = dict()
        # first, recurse
        is_add_caches = (
            all(isinstance(n, dcir.DomainLoop) for n in states) and states[0].axis == axis
        )

        if is_add_caches:
            axis = states[0].axis
            states, local_name_map = self._process_state_sequence(states, fields=fields)

        inner_name_map = dict()
        states = self.generic_visit(
            states, axis=axis, fields=fields, storage=storage, ctx_name_map=inner_name_map
        )
        local_name_map.update(inner_name_map)

        grid_subsets = self._collect_access_infos_from_memlets(states, node.field_decls)
        for k in grid_subsets.keys():
            if k not in local_name_map:
                local_name_map[k] = k

        field_decls = {decl.name: decl for decl in node.field_decls}
        memlet_names = {mem.field for mem in node.read_memlets + node.write_memlets}
        for name, cached_name in local_name_map.items():
            if cached_name in memlet_names:
                continue
            while name.startswith("__local_"):
                name = name[len("__local_") :]
            main_decl = next(decl for decl in node.field_decls if decl.name == name)

            stride = 1
            strides = []
            if cached_name.startswith("__local_"):
                shape = grid_subsets[cached_name].overapproximated_shape
            else:
                shape = grid_subsets[cached_name].shape
            for s in reversed(shape):
                strides = [stride, *strides]
                stride = f"({stride}) * ({s})"
            if is_add_caches:
                dcir_storage = dcir.StorageType.from_dace_storage(storage)
            else:
                dcir_storage = dcir.StorageType.Default
            access_info = dcir.FieldAccessInfo(
                grid_subset=grid_subsets[cached_name],
                global_grid_subset=grid_subsets[cached_name],
                dynamic_access=main_decl.access_info.dynamic_access,
                variable_offset_axes=main_decl.access_info.variable_offset_axes,
            )
            field_decls[cached_name] = dcir.FieldDecl(
                name=cached_name,
                access_info=access_info,
                dtype=main_decl.dtype,
                data_dims=main_decl.data_dims,
                strides=strides,
                storage=dcir_storage,
            )

        states = propagate_field_decls(
            states,
            decl_map={name: (decl.strides, decl.storage) for name, decl in field_decls.items()},
        )

        return dcir.NestedSDFG(
            label=node.label,
            states=states,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            field_decls=[decl for decl in field_decls.values()],
            symbol_decls=node.symbol_decls,
            name_map=node.name_map,
        )
