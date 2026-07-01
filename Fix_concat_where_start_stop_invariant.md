# `SymbolicRange` invariant: `start ≠ +inf`, `stop ≠ -inf`

`SymbolicRange.__post_init__` (in
`src/gt4py/next/iterator/ir_utils/domain_utils.py`) asserts:

```python
assert self.start is not itir.InfinityLiteral.POSITIVE
assert self.stop  is not itir.InfinityLiteral.NEGATIVE
```

## What the invariant means

Infinities are only allowed **outward**. Inward infinities are forbidden:

| form | allowed? | meaning |
|---|---|---|
| `[a, b)`, `[-inf, b)`, `[a, +inf)`, `[-inf, +inf)` | ✅ | finite / left-/right-/both-unbounded |
| `[+inf, b)`, `[a, -inf)`, `[+inf, -inf)` | ❌ | degenerate "empty from infinity" |

The union-neutral range `[+inf, -inf)` is exactly the forbidden inward form.
Empty domains are conventionally represented elsewhere with **finite** bounds
(`start >= stop`, e.g. `[10, 10)` / `[0, 0)`), never with infinities — that is
the assumption the check enforces.

## Where the assumption is relied upon

### Silent-wrong (no crash)

- **`empty()`** (`domain_utils.py`) — now classifies infinities explicitly: an
  "inward" infinity (`start is POSITIVE` or `stop is NEGATIVE`) → `True`
  (degenerate empty), an "outward" infinity (`start is NEGATIVE` or
  `stop is POSITIVE`) → `False` (always non-empty), otherwise the literal /
  equality checks as before. So the neutral `[+inf, -inf)` *is* detected as
  empty, and half-infinite ranges are no longer treated as
  statically-undecidable (which previously left `let`/`if` guard cruft in
  inferred domains).
- **`is_finite`** — reports `[+inf, -inf)` as "not finite", but it is really
  *empty*, not *unbounded*; callers gating on finiteness would mis-handle it.

### Assertion crashes (encode the invariant; in practice only hit on half-infinite condition-domains)

- **`domain_complement`** (`domain_utils.py`) —
  `assert (lb == NEGATIVE) != (ub == POSITIVE)`. For `[+inf, -inf)` both sides
  are `False` → `False != False` fails.
- **concat_where `_range_complement`** (canonicalize domain argument) —
  `assert not any(isinstance(b, itir.InfinityLiteral) for b in (start, stop))`
  requires finite bounds outright.

### Hard breakage if an inward-infinity domain reaches lowering / execution

This is the important class: a real (materialized) domain must never carry
`[+inf, -inf)`.

- **Size computed as `stop - start`**:
  - dace `gtir_domain.py:143` — `max(0, stop - start)`
  - dace `gtir_to_sdfg_scan.py:380` — `scan_domain.stop - scan_domain.start`
  - `Domain.size` / `shape` (`common.py:491`)

  With `stop = -inf`, `start = +inf` this is `-oo - (+oo)` → fragile/garbage
  sympy, and `origin = +inf` is nonsensical.
- **`assert ...is_finite(...)` on the materialized domain**: embedded
  `nd_array_field.py:189,1187` / `embedded/common.py:56`, dace
  `sdfg_callable.py:22` → assertion failure at runtime.
- **Codegen of bounds**: gtfn / dace / roundtrip now emit `InfinityLiteral`
  (`std::numeric_limits<...>::min()/max()`, `±sympy.oo`, `common.Infinity`).
  Fine for an *outward* infinity in a genuine concat_where domain, but an
  inward-infinity empty domain (`origin = +inf`, `size = -inf - +inf`) generates
  nonsensical C++ / SDFG.

## Bottom line

`ConstantFolding` itself is largely tolerant of `[+inf, -inf)`: the `plus`-only
`assert not both-infinity` never sees two infinities; `minimum` / `maximum`
fold it correctly; `greater_equal(+inf, -inf)` folds to `True`.

The blockers to allowing the inward/neutral form are therefore:
1. `empty()` won't detect it (silent),
2. the two complement assertions (crash, but not on neutral ranges in practice),
3. and — critically — it must **never escape to lowering/execution**, where
   `stop - start` and the `is_finite` asserts would break.

Today this cannot happen because the all-empty reduction returns a *finite*
empty range. A neutral-seed reduction whose inputs are all empty would instead
yield `[+inf, -inf)` and could flow downstream, so that path would need to
re-normalize all-empty results back to a finite empty range (or guarantee they
are dropped before codegen) before the invariant could be relaxed.
