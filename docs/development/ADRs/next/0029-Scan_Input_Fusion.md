# Bounded fusion of scan inputs in the DaCe backend

Status: experimental, opt-in. This change rewrites GT4Py iterator IR before
SDFG construction. It does not change the shared-output SDFG transformation,
and that separate optimization is not required.

## The pattern

A field operator prepares coefficients, then a vertical scan consumes them.
Without fusion, the generated code can write full coefficient fields to global
memory and read them again during the scan. The optimization moves eligible
coefficient arithmetic into the recurrence while preserving its level order.

The following pseudocode selects only input `a`; `b` stays outside the scan.
`producer(x)` is an upstream field computation which is not recursively fused.

```python
# Before.
p = producer(x)
a = coefficient(p)
b = other_calculation(x)
y = scan(step, initial, a, b)

# After: coefficient() is evaluated at the level that needs it.
p = producer(x)
b = other_calculation(x)
carry = initial
for k in vertical_order:
    a_value = coefficient_at_level(p, k)
    carry = step(carry, a_value, b[k])
    y[k] = carry
```

A simplified IR view makes the selection boundary explicit:

```text
Before: as_fieldop(scan(step))(as_fieldop(coefficient)(p), b)
After:  as_fieldop(scan(step_with_coefficient))(p, b)

p = as_fieldop(producer)(x) remains outside the scan.
```

These examples show the pattern, not literal GTIR syntax or a promise that every
input matches. The pass considers eligibility and safety before moving work.
An external producer result used elsewhere is not deleted merely because one
consumer can inline its expression.

## Two preparation scopes

The default `scan_fusion_scope="immediate"` fuses at most one producer layer on
each original scan input. It uses the existing `fuse_as_fieldop` helper with an
eligibility mask. Newly exposed inputs are not revisited. With no selector, all
safe immediate inputs are considered; this is not a profitability heuristic.

The optional `"field_operator"` scope first prepares local expressions before
ordinary function inlining removes their boundaries. For example:

```python
wind = prepare_wind(inputs)  # An upstream field operator stays outside.


def coefficient_stage(inputs, wind):
    gamma = local_arithmetic(inputs)
    a = expression_a(gamma, inputs)
    b = expression_b(gamma, inputs)
    return scan(step, initial, a, b, wind)
```

Local arithmetic such as `gamma` may become part of the coefficient producers.
Function parameters, calls to other field operators and other scans remain
boundaries. The later fusion still acts once on each selected immediate input,
but that input can now contain several local arithmetic operations. This is
why the preparation scope matters when interpreting “one producer layer.”
The tested ICON solver uses this scope; upstream explicit-wind preparation
stays outside its forward scan.

## Pipeline and options

The DaCe translator consumes these `optimization_args`:

| Option                | Meaning                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| `fuse_scan_inputs`    | Enable fusion; defaults to `False` and requires auto-optimization.                                  |
| `scan_input_selector` | Optional `(scan_call, input_index) -> bool`; indices refer to original inputs, excluding the carry. |
| `scan_fusion_scope`   | `"immediate"` by default; `"field_operator"` enables the preparation described above.               |

A selector returning `input_index == 0` chooses only the first eligible input.
It cannot override the safety checks. Prefer a top-level, picklable function for
compilation in worker processes. With field-operator preparation enabled, the
selector sees the resulting normalized scan.

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

The call path is:

```text
normalize_scan_producers          # Only for field_operator scope.
    -> apply_fieldview_transforms
    -> fuse_scan_inputs
         -> select eligible immediate inputs and check output overlap
         -> existing fuse_as_fieldop helper
         -> lift supported vertical shifts to scalar-valued scan arguments
         -> type inference and apply_fieldview_transforms after a rewrite
    -> build_sdfg_from_gtir
    -> normal SDFG optimization and code generation
```

Read [scan_fusion.py](../../../../src/gt4py/next/program_processors/runners/dace/scan_fusion.py)
and its call sites in
[translation.py](../../../../src/gt4py/next/program_processors/runners/dace/workflow/translation.py).
The preparation and normal pre-fusion field-view pipeline respect
`disable_itir_transforms`; the explicit scan-fusion option has its own guard.

## Safety and limits

Scan direction, initial state and arithmetic casts are retained. The overlap
check follows symbols and let aliases to reject reads that overlap assignment
targets. It does not detect differently named fields backed by overlapping
runtime buffers: callers enabling the pass must not supply those aliases.

DaCe scan arguments are values at the current level. `_LiftScanShifts` moves
supported fixed vertical shifts into separate field arguments. Unsupported
shifted accesses leave the original scan unchanged. A successful rewrite runs
normal field-view transforms again, preserving the offset provider and domain
policy. Fusion is not a guarantee of lower register use or faster execution.

[test_scan_fusion.py](../../../../tests/next_tests/unit_tests/program_processor_tests/runners_tests/dace_tests/test_scan_fusion.py)
covers forward/backward scans, compiled values, selected inputs, non-recursion,
upstream boundaries, symbolic/let aliasing, unsupported shifts, shared
coefficients and both field-view pipeline calls. In particular, read
`test_selected_input_only_and_no_recursive_producer_fusion`,
`test_field_operator_scope_preserves_upstream_boundary` and
`test_fieldview_pipeline_runs_before_and_after_fusion`.

## When this helps

The useful case has coefficient fields whose materialization costs more than
computing their values inside the recurrence. In the inspected ICON solvers,
each specialization lost one kernel and three temporary arrays. This code
inspection does not by itself quantify HBM bytes or explain the GPU timing.

The latest combined-pass experiment measured total solver-time reductions of
12.48% regional and 11.64% global on MI300A, and 4.73% regional and 11.39% global
on GH200. Those are program contributions within a combined treatment, not
standalone measurements of this PR. Whole-granule gains also include the separate
shared-output optimization. Earlier direct comparisons found no resolved
performance difference between the compiler solver and the frontend prototype;
the compiler implementation replaces that prototype's benefit.

See the [ICON results and measurement definitions](https://github.com/dganellari/icon4py/blob/e1a3db855/amd_scripts/docs/GLOBAL_REVIEW.md).
The GPU-tested combined GT4Py revision was `403f9d99`; this branch preserves its
scan implementation and tests. That stack also included the independent
array-scalar warning fix, needed when GPU domain inference emits the warning.
No general profitability rule or default activation is proposed. This review
targets `amd_chiplet_setting`; merging there does not deliver the change to `main`.
