# Guarded shared-output and scan-input fusion in the DaCe backend

Status: experimental, opt-in.

## Context

A producer and consumer can be separated by avoidable global temporary storage
or GPU launches. Two existing fusion mechanisms cover relevant cases, but need
explicit selection and safety guards: vertical map splitting through an external
output, and field-operator fusion into a scan. Their implementation must preserve
model equations and avoid absorbing arbitrarily long upstream expression chains.

## Decision

Reuse the existing transformations. Keep both extensions disabled by default,
with independent selection. Do not introduce model-specific names or a universal
profitability rule into GT4Py.

### Shared-output map splitting

`VerticalSplitMapRange.allow_shared_data` permits splitting a non-transient array
only when its writers form a disjoint, pointwise partition and each reader is
covered by exactly one writer. Unknown overlap, aliased arrays, views, shifted or
crossing reads, reductions, dynamic accesses and unsupported map structures are
rejected. External descriptors and stores remain intact, including values read
by subsequent states. Splitting access nodes removes false graph dependencies;
it does not remove the externally visible writes.

Use `gt_vertical_map_split_fusion(..., allow_shared_data=True)` for a direct
transformation. The standard auto-optimizer can select individual candidates
through its existing callback:

```python
from gt4py.next.program_processors.runners.dace import transformations


def select_shared_output(transformation, first_map, second_map, graph, sdfg):
    transformation.allow_shared_data = transformation.access_node.data == "shared_output"
    return True


optimization_hooks = {
    transformations.GT4PyAutoOptHook.TopLevelDataFlowVerticalSplitCallBack: select_shared_output,
}
```

Pass these hooks through the backend's `optimization_hooks` option. The name
above is an example selection by the application, not a built-in GT4Py rule.

### Scan-input fusion

The DaCe translator consumes these `optimization_args` before SDFG lowering:

- `fuse_scan_inputs`: boolean, default `False`; requires auto-optimization.
- `scan_input_selector`: optional callable `(scan_call, input_index) -> bool`.
  Indices exclude the scan carry. It narrows eligibility, never overrides safety.
  Prefer a top-level, picklable callable for compilation in worker processes.
- `scan_fusion_scope`: `"immediate"` by default, or `"field_operator"`.

For example:

```python
from gt4py.next.program_processors.runners.dace.workflow.backend import make_dace_backend

backend = make_dace_backend(
    gpu=True,
    optimization_args={
        "fuse_scan_inputs": True,
        "scan_fusion_scope": "field_operator",
    },
)
```

The default scope calls the existing low-level `fuse_as_fieldop` helper once per
original scan, using an eligibility mask. Newly exposed producers are not
revisited. Without a selector, all safe immediate inputs are considered; enabling
the pass does not mean selecting one particular input. For example, a selector
returning `input_index == 0` restricts fusion to the first eligible input.

The optional `"field_operator"` scope first normalizes local producer expressions
in functions that call simple scan wrappers, before function inlining erases
those boundaries. Local arithmetic lets can be expanded, including shared
coefficient expressions. Function parameters and calls to other field operators
or scans remain boundaries. This can form a complete coefficient producer from
several local operations; it deliberately permits more than one raw IR layer
inside that original function. The selector then sees the normalized scan.

The actual scan fusion still occurs once per scan. It preserves direction,
initial state and casts; follows let aliases when checking overlap with output
targets; and lifts supported fixed vertical shifts to field arguments, respecting
DaCe's scalar scan-argument convention. Unsupported accesses leave the original
scan unchanged. The translator runs normal `apply_fieldview_transforms` before
fusion, and the pass runs it again after a successful rewrite with the original
offset provider and domain policy.

## Consequences and validation

Shared outputs remain externally observable. Scan fusion is bounded by explicit
producer/function scope rather than a recursive driver. The two transformations
remain independent and testable without ICON model sources.

Tests cover numerical results, untouched domains, external and later-state
outputs, rejected unsafe map partitions, forward/backward recurrences, selected
inputs, upstream boundaries, aliasing, shifted inputs, shared coefficients and
both field-view pipeline calls.

In one regional/120 ICON solve_nonhydro comparison per vendor, combining the two
compiler transformations reduced summed program device time by 5.99% on MI300A
and 2.06% on GH200; MI300A granule wall time fell 4.55%. All 148 checked fields
matched exactly. Direct comparisons with the earlier frontend solver rewrite
showed no resolved difference. These results motivate opt-in use, not a general
speedup guarantee or a default policy. Global-grid regression testing remains
outstanding; cache capacity is not established as the cause of the vendor gap.
