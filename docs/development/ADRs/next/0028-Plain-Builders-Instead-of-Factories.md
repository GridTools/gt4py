---
tags: [backend, otf, toolchain, workflows, dependencies]
---

# Plain Builders Instead of factory-boy Factories

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-08-20
- **Updated**: 2026-08-20

In the context of composing the GTFN and DaCe backends and their OTF compile
workflows, facing a production dependency on `factory-boy` — a test-data
library — whose `Trait` / `SubFactory` / `SelfAttribute` / `LazyAttribute`
machinery and stringly-typed `__`-path overrides are invisible to the type
checker, we decided to replace the factory classes with plain builder
functions over the existing frozen dataclasses, and to validate injected
sub-components explicitly, to achieve statically checked composition, one
fewer runtime dependency, and loud failures where the factories failed
silently.

## Context

Every object these factories build — `Backend`, `OTFCompileWorkflow`,
`GTFNTranslationStep`, `DaCeTranslator`, the compilers — is already a frozen
dataclass. `factory-boy` added a second, parallel construction language on top:

- **Untyped.** The declarations are class attributes of a `Params` block, so
  `mypy` cannot check them. `src/` carried **8** \`# type: ignore[assignment]

  # factory-boy typing not precise enough\` suppressions solely to keep the

  factories quiet.

- **Silently wrong.** Overrides are `__`-delimited strings resolved at
  runtime. When a path does not resolve, nothing happens. This was not
  hypothetical: `run_gtfn_imperative` was declared as

  ```python
  run_gtfn_imperative = GTFNBackendFactory(
      name_postfix="_imperative",
      otf_workflow__translation__use_imperative_backend=True,
  )
  ```

  but the `cached_translation` trait replaces `translation` with a
  `CachedStep`, so the path never reached the wrapped `GTFNTranslationStep`.
  The backend had `use_imperative_backend=False` — it was the declarative
  backend under another name, and the `GTFN_CPU_IMPERATIVE` entry of the test
  matrix had therefore never exercised imperative code generation.
  `run_gtfn_no_transforms` was likewise named `run_gtfn_cpu`, colliding with
  `run_gtfn`.

- **A runtime dependency for a test-time concern.** `factory-boy` sat in
  `[project] dependencies`, shipped to every user, to compose four backends.

## Decision

Factory classes are replaced by **plain builder functions**; `factory-boy`
moves to the `test` dependency group, where the `cartesian` and `eve` IR
test-data factories keep using it for what it is designed for.

Builders follow two rules:

1. **A builder takes cross-cutting configuration only** — device, caching,
   build type, auto-optimize — and uses it to configure the steps it creates.
2. **An injected sub-component is used verbatim.** A caller that wants a
   different translation step builds one and passes it; the builder never
   reaches into it to stamp fields onto it.

```python
run_gtfn_imperative = make_gtfn_backend(
    name_postfix="_imperative",
    translation=gtfn_module.GTFNTranslationStep(use_imperative_backend=True),
)
```

Rule 2 creates one hazard: an injected step could disagree with the
cross-cutting configuration — a CPU translation step in a GPU toolchain.
`workflow.check_device_agreement(step, device_type, what)` turns that into a
`ValueError` at construction time. It inspects only steps that structurally
declare a device (the `workflow.DeviceConfigurable` protocol) and is used to
**check**, never to mutate.

We considered a `with_changes(step, **changes)` helper that stamps
cross-cutting fields onto whichever component is present, applying only the
fields the target declares. We rejected it: silently ignoring the fields a
target does not declare reproduces exactly the failure mode that motivated
this ADR — the `run_gtfn_imperative` bug is what a silent no-op looks like
after a year. Checking is the same amount of introspection with the opposite
failure mode.

Builder defaults preserve the previous factory semantics: a standalone
compile-workflow builder leaves translation caching **off** (the
`cached_translation` trait was opt-in), while the backend builders turn it on.

## Consequences

- Composition is ordinary, statically checked Python. The 8 factory-related
  `type: ignore` suppressions are gone, and a misspelled parameter is now a
  `TypeError` at import rather than a silently ignored override.
- One fewer runtime dependency.
- **`run_gtfn_imperative` now actually uses the imperative backend.** This is
  a behaviour change: the `GTFN_CPU_IMPERATIVE` test-matrix entry begins
  exercising imperative code generation for the first time, and it
  immediately fails on the pre-existing IR defect tracked in issue #2810.
  Two call sites are xfailed against that issue (`test_hdiff` and
  `test_concat_where::test_lap_like[static_domains]`); fixing the defect is
  out of scope for a construction refactor.
- **`run_gtfn_no_transforms` is renamed** from `run_gtfn_cpu` to
  `run_gtfn_cpu_no_transforms`, removing the collision with `run_gtfn`. No
  cache is affected: the build cache keys on the entry-point name plus a
  fingerprint of the `ExtensionSource`, and the translation-cache directory
  is keyed on the literal backend family (`gtfn` / `dace`). `Backend.name`
  reaches only the metrics source key and one error message, so what the
  collision actually cost was two distinct backends sharing one metrics
  identity.
- All other pre-built backends are unchanged, verified field-by-field against
  the previous construction.
- Customizing a single knob of a sub-component now means building that
  component, rather than passing a `__`-path string. This is more explicit and
  slightly more verbose; `make_dace_backend` keeps its translator-local
  keyword arguments so existing external callers are unaffected.
