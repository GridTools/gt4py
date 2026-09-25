# Guarded fusion through shared outputs

Status: experimental, opt-in. This change acts on DaCe SDFGs; it does not
change GTIR or scan lowering. The separate scan-input fusion is not required.

## The pattern

A producer computes two external outputs over a vertical column. Two consumers
read one output over different vertical bands. The producer range does not
match either consumer, so it must be split before the maps can be fused.
Previously, the shared non-transient output prevented that split.

This is the pattern used in `test_shared_output_split_preserves_values`.
Ranges below are half-open, and both consumers are present in both versions.
The example omits the unchanged horizontal loop `i in range(1, 5)`.

```python
# Before: one producer map and two consumer maps.
for k in range(12):
    theta[i, k] = input[i, k] + 1
    rho[i, k] = input[i, k] * 2
for k in range(3):
    wind[i, k] = theta[i, k] * 3
for k in range(3, 12):
    wind[i, k] = theta[i, k] * 4

# After splitting the producer and fusing compatible pieces.
for k in range(3):
    value = input[i, k] + 1
    theta[i, k] = value
    rho[i, k] = input[i, k] * 2
    wind[i, k] = value * 3
for k in range(3, 12):
    value = input[i, k] + 1
    theta[i, k] = value
    rho[i, k] = input[i, k] * 2
    wind[i, k] = value * 4
```

The relevant part of the SDFG is schematically:

```text
Before                               After

Producer map [0:12]                  Fused map [0:3]
  | theta [0:12]                       | theta [0:3], rho [0:3], wind [0:3]
  +--> consumer map [0:3]             Fused map [3:12]
  +--> consumer map [3:12]              | theta [3:12], rho [3:12], wind [3:12]
  | rho [0:12]

External theta and rho stores remain in both graphs.
A following state can still read the complete external theta field.
```

This is a schematic, not a generated SDFG capture. The transformation splits
access paths and removes dependencies that unnecessarily prevent fusion;
it does not make external outputs temporary or remove their stores.
The `upper_consumer` test parameter changes the input graph, so it must stay
fixed when comparing transformation options. It is not an optimization flag.

## Implementation and selection

Read `VerticalSplitMapRange` in
[map_fusion_extended.py](../../../../src/gt4py/next/program_processors/runners/dace/transformations/map_fusion_extended.py),
starting with `_shared_access_partition`, then `_split_shared_access` and their
call sites. The existing map-fusion machinery performs the subsequent fusion.

`allow_shared_data` defaults to `False`. For direct use, enable it through
`gt_vertical_map_split_fusion(..., allow_shared_data=True)`. The
`fuse_map_fragments` option controls fusion of the fragments created by the split.
For normal backend optimization, pass
`optimization_args={"vertical_split_allow_shared_data": True}`. This defaults to
`False`, so applications can enable it per program without inspecting SDFG nodes.
The compiler still checks every candidate for safe partitions. This enables all
safe shared-output candidates in the selected program, which is broader than a
callback restricted to one output name; GPU performance must be checked for
that configuration. To limit candidates by their connecting array, also pass
`vertical_split_shared_data=("shared_output",)`. An empty sequence allows no
shared-output candidates; `None` allows any name. This selection does not change
the treatment of transient arrays or bypass the safety checks. It lets an
application retain a measured field selection without inspecting SDFG nodes.
Use `vertical_split_shared_data_dimension=KDim` to allow splitting only along
that dimension while requiring all other map ranges to match. This avoids
splitting a shared producer across both horizontal and vertical boundaries when
only the vertical split has been measured. Both restrictions leave transient
candidates unchanged.

For ICON theta-rho, selecting its theta output and K dimension reproduced the
old restricted callback's generated GPU source in eight local comparisons:
regional/global, AMD/NVIDIA and IAU off/on. These checks used 120 levels and
representative static cutoffs, without GPU compilation or execution. They
preserve the inspected graph shape; they are not new timing measurements.

For finer selection, use the existing
`TopLevelDataFlowVerticalSplitCallBack` hook:

```python
from gt4py.next.program_processors.runners.dace import transformations


def select_shared_output(transformation, first_map, second_map, graph, sdfg):
    transformation.allow_shared_data = transformation.access_node.data == "shared_output"
    return True


optimization_hooks = {
    transformations.GT4PyAutoOptHook.TopLevelDataFlowVerticalSplitCallBack: select_shared_output,
}
```

Pass this dictionary through the backend's `optimization_hooks` option. The
example assigns the flag for every candidate because the matcher reuses the
transformation object. A callback selects a candidate; it cannot bypass the
transformation's safety checks. The output name is an application choice, not
a built-in GT4Py rule.

## Safety and limits

Accepted writers form disjoint, pointwise, unit-stride partitions, with outer
subsets equal to map ranges. Each reader must be covered by exactly one writer.
The bounding-box `covers()` test is exact only under those guards. Unknown
overlap, arrays marked `may_alias`, views, shifted or crossing reads, reductions,
dynamic accesses and unsupported map structures are rejected. External array
descriptors and stores, including values used in later states, remain intact.
There is no runtime pointer-alias check.

Tests in
[test_vertical_map_split_shared.py](../../../../tests/next_tests/unit_tests/program_processor_tests/runners_tests/dace_tests/transformation_tests/test_vertical_map_split_shared.py)
cover compiled values, untouched regions, later-state reads, rejected partitions
and callback selection. Keep the existing vertical-map-split/fusion tests in
the review as regression coverage for the unchanged transient-only path.

## When this helps

The opportunity is a producer whose range can be partitioned to match its
consumers while preserving all external writes. This can avoid an intermediate
read and reduce launches; actual code generation and performance still need
checking. More splitting is not automatically faster.

In the tested ICON regional theta-rho program, the combined compiler configuration
changed six kernels to five; global retained three. The latest joint experiment
measured a 13.83% regional theta-rho time reduction on MI300A and 2.63% on GH200.
These are program contributions with both compiler passes enabled, not isolated
measurements of this pass. The whole-granule gains also include the separate
solver optimization and must not be credited to this PR alone.

See the [ICON results and measurement definitions](https://github.com/dganellari/icon4py/blob/e1a3db855/amd_scripts/docs/GLOBAL_REVIEW.md).
The GPU-tested combined GT4Py revision was `403f9d99`; this branch preserves its
shared-output implementation and tests. The GPU stack also included the separate
array-scalar warning fix. No general profitability rule or default activation is
proposed. This review targets `amd_chiplet_setting`; merging there does not
by itself deliver the change to `main`.
