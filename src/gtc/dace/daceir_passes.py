from typing import Any, Dict, List, Tuple

import dace

import eve
from eve.iterators import iter_tree
from gtc import common
from gtc import daceir as dcir
from gtc import oir

from .utils import union_node_access_infos


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


def set_idx(axis, access_infos, idx):
    res = dict()
    for name, info in access_infos.items():
        grid_subset = set_grid_subset_idx(axis, info.grid_subset, idx)
        res[name] = dcir.FieldAccessInfo(
            grid_subset=grid_subset,
            global_grid_subset=info.global_grid_subset,
            dynamic_access=info.dynamic_access,
            variable_offset_axes=info.variable_offset_axes,  # should actually never be True here
        )
    return res


class FieldAccessRenamer(eve.NodeMutator):
    def apply(self, node, *, local_name_map):
        return self.visit(node, local_name_map=local_name_map)

    def _rename_accesses(self, access_infos, *, local_name_map):
        return {
            local_name_map[name] if name in local_name_map else name: info
            for name, info in access_infos.items()
        }

    def visit_CopyState(self, node: dcir.CopyState, *, local_name_map):
        name_map = {local_name_map[k]: local_name_map[v] for k, v in node.name_map.items()}
        return dcir.CopyState(
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
            name_map=name_map,
        )

    def visit_DomainMap(self, node: dcir.DomainMap, *, local_name_map):
        computations = self.visit(node.computations, local_name_map=local_name_map)
        return dcir.DomainMap(
            index_ranges=node.index_ranges,
            computations=computations,
            schedule=node.schedule,
            grid_subset=node.grid_subset,
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
        )

    def visit_DomainLoop(self, node: dcir.DomainLoop, *, local_name_map):
        return dcir.DomainLoop(
            axis=node.axis,
            index_range=node.index_range,
            loop_states=self.visit(node.loop_states, local_name_map=local_name_map),
            grid_subset=node.grid_subset,
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
        )

    def visit_StateMachine(self, node: dcir.StateMachine, *, local_name_map: Dict[str, str]):
        name_map = dict(node.name_map)
        for old_array_name in node.name_map.keys():
            if old_array_name in local_name_map:
                new_array_name = local_name_map[old_array_name]
                name_map[new_array_name] = name_map[old_array_name]
                del name_map[old_array_name]
        return dcir.StateMachine(
            label=node.label,
            field_decls=node.field_decls,  # don't rename, this is inside
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
            symbols=node.symbols,
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
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
            name_map=name_map,
            stmts=node.stmts,
        )


rename_field_accesses = FieldAccessRenamer().apply


class FieldDeclPropagator(eve.NodeMutator):
    def apply(self, node, *, decl_map):
        return self.visit(node, decl_map=decl_map)

    def visit_StateMachine(
        self,
        node: dcir.StateMachine,
        *,
        decl_map: Dict[str, Tuple[Any, dace.StorageType]],
    ):
        field_decls = dict(node.field_decls)
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
        return dcir.StateMachine(
            label=node.label,
            field_decls=field_decls,  # don't rename, this is inside
            read_accesses=node.read_accesses,
            write_accesses=node.write_accesses,
            symbols=node.symbols,
            states=states,
            name_map=node.name_map,
        )


propagate_field_decls = FieldDeclPropagator().apply


class MakeLocalCaches(eve.NodeTranslator):
    def _write_before_read_fields(self, axis, node):
        for state_machine in iter_tree(node).if_isinstance(dcir.StateMachine):
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
            elif isinstance(node, oir.MaskStmt):
                yield from _iterate_access_order(node.mask, is_write=False, is_masked=is_masked)
                yield from _iterate_access_order(node.body, is_masked=True)
            elif isinstance(node, oir.AssignStmt):
                yield from _iterate_access_order(node.right, is_write=False, is_masked=is_masked)
                yield from _iterate_access_order(node.left, is_write=True, is_masked=is_masked)
            else:
                for node in iter_tree(node).if_isinstance(common.FieldAccess):
                    yield node, is_write, is_masked

        for acc, is_write, is_masked in _iterate_access_order(tasklets):
            if is_write and not is_masked:
                if acc.name not in maybe_read_first_fields:
                    guaranteed_write_first_fields.add(acc.name)
            elif not is_write and acc.name not in guaranteed_write_first_fields:
                maybe_read_first_fields.add(acc.name)

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

    def _set_access_info_subset(self, access_infos, subsets):
        res = dict(access_infos)
        for name, subset in subsets.items():
            if name in access_infos:
                access_info = access_infos[name]
                res[name] = dcir.FieldAccessInfo(
                    grid_subset=subset,
                    global_grid_subset=access_info.global_grid_subset,
                    dynamic_access=access_info.dynamic_access,
                    variable_offset_axes=access_info.variable_offset_axes,
                )
        return res

    def _make_localcache_states(
        self,
        loop,
        loop_states,
        *,
        context_read_accesses,
        context_write_accesses,
        context_field_accesses,
        local_name_map,
        cache_subsets,
    ):

        # also, init
        from .utils import union_node_access_infos, union_node_grid_subsets

        grid_subset = union_node_grid_subsets(list(loop_states))
        read_accesses, write_accesses, field_accesses = union_node_access_infos(list(loop_states))
        axis = loop.axis

        cache_read_subsets = self._get_cache_read_subset(axis, loop_states, local_name_map)
        fill_subsets = self._get_fill_subsets(
            axis, loop_states, loop.index_range.stride, local_name_map
        )
        flush_subsets = {
            name: info.grid_subset
            for name, info in write_accesses.items()
            if name in local_name_map
        }
        init_subsets = self._subsets_diff(
            axis, self._subsets_diff(axis, cache_read_subsets, cache_subsets), fill_subsets
        )

        shift_subsets = self._subsets_diff(axis, cache_read_subsets, fill_subsets)
        cache_subsets.clear()
        cache_subsets.update(
            {k: v for k, v in shift_subsets.items() if v is not None},
        )

        def set_local_names(access_infos):
            return {local_name_map[k]: v for k, v in access_infos.items()}

        def shift(access_infos, offset):
            res = dict()
            for name, info in access_infos.items():
                grid_subset = info.grid_subset.set_interval(
                    axis, info.grid_subset.intervals[axis].shifted(offset)
                )
                res[name] = dcir.FieldAccessInfo(
                    grid_subset=grid_subset,
                    global_grid_subset=info.global_grid_subset,
                    dynamic_access=info.dynamic_access,
                    variable_offset_axes=info.variable_offset_axes,  # should actually never be True here
                )
            return res

        init_accesses = self._set_access_info_subset(
            {
                k: v
                for k, v in field_accesses.items()
                if k in init_subsets and init_subsets[k] is not None
            },
            init_subsets,
        )
        init_state = dcir.CopyState(
            grid_subset=grid_subset,
            read_accesses=set_idx(axis, init_accesses, loop.index_range.start),
            write_accesses=set_local_names(init_accesses),
            name_map=local_name_map,
        )

        fill_accesses = self._set_access_info_subset(
            {
                k: v
                for k, v in field_accesses.items()
                if k in fill_subsets and fill_subsets[k] is not None
            },
            fill_subsets,
        )
        fill_state = dcir.CopyState(
            grid_subset=grid_subset,
            read_accesses=fill_accesses,
            write_accesses=set_local_names(fill_accesses),
            name_map=local_name_map,
        )
        loop_states = rename_field_accesses(loop_states, local_name_map=local_name_map)

        flush_accesses = self._set_access_info_subset(
            {
                k: v
                for k, v in field_accesses.items()
                if k in flush_subsets and flush_subsets[k] is not None
            },
            flush_subsets,
        )
        flush_state = dcir.CopyState(
            grid_subset=grid_subset,
            read_accesses=set_local_names(flush_accesses),
            write_accesses=flush_accesses,
            name_map={v: k for k, v in local_name_map.items()},
        )

        shift_accesses = self._set_access_info_subset(
            {
                k: v
                for k, v in field_accesses.items()
                if k in shift_subsets and shift_subsets[k] is not None
            },
            shift_subsets,
        )
        shift_state = dcir.CopyState(
            grid_subset=grid_subset,
            read_accesses=shift(set_local_names(shift_accesses), loop.index_range.stride),
            write_accesses=set_local_names(shift_accesses),
            name_map={v: v for v in local_name_map.values()},
        )
        return init_state, (fill_state, *loop_states, flush_state, shift_state)

    def _process_state_sequence(self, states, *, localcache_infos):

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
                    name in localcache_infos.fields
                    and isinstance(interval, dcir.IndexWithExtent)
                    and interval.size > 1
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
                    last_loop_node.index_range.end == loop_node.index_range.start
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
                        read_accesses=loop_node.read_accesses,
                        write_accesses=loop_node.write_accesses,
                    ),
                ]
            )
            last_loop_node = loop_node
        return res_states, local_name_map

    def visit_DomainLoop(self, node: dcir.DomainLoop, *, ctx_name_map, localcache_infos):
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
            if loop_nodes[0].axis in localcache_infos:
                domain_loop_nodes, local_name_map = self._process_state_sequence(
                    loop_nodes, localcache_infos=localcache_infos[loop_nodes[0].axis]
                )
                res_loop_states = [*loop_states[:start_idx], *domain_loop_nodes]
                if end_idx:
                    res_loop_states += loop_states[end_idx:]
                loop_states = res_loop_states
                ctx_name_map.update(local_name_map)

        inner_name_map = dict()
        loop_states = self.generic_visit(
            loop_states, localcache_infos=localcache_infos, ctx_name_map=inner_name_map
        )
        ctx_name_map.update(inner_name_map)

        from .utils import union_node_access_infos

        read_accesses, write_accesses, _ = union_node_access_infos(loop_states)
        return dcir.DomainLoop(
            grid_subset=node.grid_subset,
            read_accesses=read_accesses,
            write_accesses=write_accesses,
            axis=node.axis,
            index_range=node.index_range,
            loop_states=loop_states,
        )

    def visit_StateMachine(self, node: dcir.StateMachine, *, localcache_infos, **kwargs):
        states = node.states
        local_name_map = dict()
        # first, recurse
        is_add_caches = (
            all(isinstance(n, dcir.DomainLoop) for n in states)
            and states[0].axis in localcache_infos
        )

        if is_add_caches:
            axis = states[0].axis
            states, local_name_map = self._process_state_sequence(
                states, localcache_infos=localcache_infos[states[0].axis]
            )
        from .utils import union_node_access_infos

        inner_name_map = dict()
        states = self.generic_visit(
            states, localcache_infos=localcache_infos, ctx_name_map=inner_name_map
        )
        local_name_map.update(inner_name_map)

        _, _, inner_field_accesses = union_node_access_infos(
            [s for loop in states if isinstance(loop, dcir.DomainLoop) for s in loop.loop_states]
        )
        field_accesses = {
            k: v if k.startswith("__local_") else node.field_decls[k].access_info
            for k, v in inner_field_accesses.items()
        }
        for k in field_accesses.keys():
            if k not in local_name_map:
                local_name_map[k] = k

        field_decls = dict(node.field_decls)
        for name, cached_name in local_name_map.items():
            if cached_name in node.read_accesses or cached_name in node.write_accesses:
                continue
            while name.startswith("__local_"):
                name = name[len("__local_") :]
            main_decl = node.field_decls[name]

            stride = 1
            strides = []
            if cached_name.startswith("__local_"):
                shape = field_accesses[cached_name].overapproximated_shape
            else:
                shape = field_accesses[cached_name].shape
            for s in reversed(shape):
                strides = [stride, *strides]
                stride = f"({stride}) * ({s})"
            if is_add_caches:
                storage = dcir.StorageType.from_dace_storage(localcache_infos[axis].storage)
            else:
                storage = dcir.StorageType.Default
            field_decls[cached_name] = dcir.FieldDecl(
                name=cached_name,
                access_info=field_accesses[cached_name],
                dtype=main_decl.dtype,
                data_dims=main_decl.data_dims,
                strides=strides,
                storage=storage,
            )

        states = propagate_field_decls(
            states,
            decl_map={name: (decl.strides, decl.storage) for name, decl in field_decls.items()},
        )

        return dcir.StateMachine(
            label=node.label,
            states=states,
            read_accesses=node.read_accesses,
            write_accesses=node.write_accesses,
            field_decls=field_decls,
            symbols=node.symbols,
            name_map=node.name_map,
        )
