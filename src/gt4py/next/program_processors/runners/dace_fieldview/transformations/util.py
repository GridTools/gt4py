# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality for the transformations/optimization pipeline."""

import dace


def gt_make_transients_persistent(
    sdfg: dace.SDFG,
    device: dace.DeviceType,
) -> dict[int, set[str]]:
    """
    Changes the lifetime of certain transients to `Persistent`.

    A persistent lifetime means that the transient is allocated only the very first
    time and only deallocated if the underlying `CompiledSDFG` object goes out of
    scope or if the exit handler of the SDFG is called. The main advantage is,
    that memory must not be allocated, however, the SDFG can not be called by
    different threads.

    Args:
        sdfg: The SDFG to process.
        device: The device type.

    Returns:
        A dictionary mapping SDFG IDs to a set of transient arrays that
        were made persistent.

    Notes:
        This function was copied from DaCe. Furthermore, the DaCe version does
        also resets the `wcr_nonatomic` property, i.e. makes every reduction
        atomic. However, this is only done for GPU and for the top level.
        This function does not do this.
    """
    result: dict[int, set[str]] = {}
    for nsdfg in sdfg.all_sdfgs_recursive():
        fsyms: set[str] = nsdfg.free_symbols
        modify_lifetime: set[str] = set()
        not_modify_lifetime: set[str] = set()

        for state in nsdfg.states():
            for dnode in state.data_nodes():
                if dnode.data in not_modify_lifetime:
                    continue

                if dnode.data in nsdfg.constants_prop:
                    not_modify_lifetime.add(dnode.data)
                    continue

                desc = dnode.desc(nsdfg)
                if not desc.transient or type(desc) not in {dace.data.Array, dace.data.Scalar}:
                    not_modify_lifetime.add(dnode.data)
                    continue
                if desc.storage == dace.StorageType.Register:
                    not_modify_lifetime.add(dnode.data)
                    continue

                if desc.lifetime == dace.AllocationLifetime.External:
                    not_modify_lifetime.add(dnode.data)
                    continue

                try:
                    # The symbols describing the total size must be a subset of the
                    #  free symbols of the SDFG (symbols passed as argument).
                    # NOTE: This ignores the renaming of symbols through the
                    #   `symbol_mapping` property of nested SDFGs.
                    if not set(map(str, desc.total_size.free_symbols)).issubset(fsyms):
                        not_modify_lifetime.add(dnode.data)
                        continue
                except AttributeError:  # total_size is an integer / has no free symbols
                    pass

                # Make it persistent.
                modify_lifetime.add(dnode.data)

        # Now setting the lifetime.
        result[nsdfg.cfg_id] = modify_lifetime - not_modify_lifetime
        for aname in result[nsdfg.cfg_id]:
            nsdfg.arrays[aname].lifetime = dace.AllocationLifetime.Persistent

    return result
