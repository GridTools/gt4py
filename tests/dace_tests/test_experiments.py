import dace
import numpy as np


def unstructured_shift(source_field: np.array, target_to_source_map: np.array) -> np.array:
    target_size = target_to_source_map.shape[0]
    num_neighbors = target_to_source_map.shape[1]
    target_field = np.zeros((target_size, num_neighbors))
    for target_element in range(0, target_size):
        for neighbor in range(0, num_neighbors):
            target_field[target_element, neighbor] = source_field[target_to_source_map[target_element, neighbor]]
    return target_field


def unstructured_shift_dace(source_field: np.array, target_to_source_map: np.array) -> np.array:
    num_targets = target_to_source_map.shape[0]
    num_neighbors = target_to_source_map.shape[1]
    target_field = np.zeros((num_targets, num_neighbors), dtype=source_field.dtype)

    sdfg = dace.SDFG(name="unstructured_shift")
    state = sdfg.add_state("state", True)

    source_shape = (
        dace.symbol("num_sources", dtype=dace.int64),
    )
    map_shape = (
        dace.symbol("num_targets", dtype=dace.int64),
        dace.symbol("num_neighbors", dtype=dace.int64),
    )

    sdfg.add_array("source_field", shape=source_shape, dtype=source_field.dtype.type)
    sdfg.add_array("target_to_source_map", shape=map_shape, dtype=target_to_source_map.dtype.type)
    sdfg.add_array("target_field", shape=map_shape, dtype=target_field.dtype.type)

    domain = {"idx_target": "0:num_targets", "idx_neighbors": "0:num_neighbors"}

    input_memlets = {
        "source_field_whole": dace.Memlet(
            data="source_field",
            subset="0:num_sources",
        ),
        "target_to_source_map_element": dace.Memlet(
            data="target_to_source_map",
            subset="idx_target, idx_neighbors",
        ),
    }

    output_memlets = {
        "target_field_element": dace.Memlet(
            data="target_field",
            subset="idx_target, idx_neighbors",
        )
    }

    tasklet_code = "target_field_element = source_field_whole[target_to_source_map_element]"

    state.add_mapped_tasklet(
        name="ushift",
        map_ranges=domain,
        inputs=input_memlets,
        code=tasklet_code,
        outputs=output_memlets,
        external_edges=True,
        schedule=dace.ScheduleType.Sequential,
    )

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "build_type", value="Debug")
        dace.config.Config.set("compiler", "cpu", "args", value="-O0")
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg(
            source_field=source_field,
            target_to_source_map=target_to_source_map,
            target_field=target_field,
            num_neighbors=num_neighbors,
            num_sources=source_field.shape[0],
            num_targets=num_targets
        )

    sdfg.view()

    return target_field


def test_unstructured_shift():
    source_field = np.array([1, 2, 3], dtype=np.float32)
    mapping = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 1],
        ],
        dtype=np.int64
    )

    expected = unstructured_shift(source_field, mapping)
    dace_result = unstructured_shift_dace(source_field, mapping)
    assert np.allclose(expected, dace_result)


arr = [1, 3, 2, 6]
out = [1, 4, 6, 12]


def partial_sum(data: np.array) -> np.array:
    n = data.shape[0]
    k = data.shape[1]

    out = np.zeros_like(data)

    init = 0
    reducer = lambda acc, elem: (acc + elem, acc + elem)

    for n_idx in range(0, n):  # parallel
        acc = init
        for k_idx in range(0, k):  # sequential
            data_element = data[n_idx, k_idx]
            result, acc = reducer(acc, data_element)
            out[n_idx, k_idx] = result

    return out


def test_partial_sum():
    source_field = np.array([1, 2, 3], dtype=np.float32)
    mapping = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 1],
        ],
        dtype=np.int64
    )

    expected = unstructured_shift(source_field, mapping)
    dace_result = unstructured_shift_dace(source_field, mapping)
    assert np.allclose(expected, dace_result)