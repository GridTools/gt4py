---
tags: [backend, otf, toolchain, workflows, dependencies]
---

# Plain Builders Instead of factory-boy Factories

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-08-20
- **Updated**: 2026-09-23

In the context of composing the GTFN and DaCe toolchains and their OTF compile
workflows, facing a production dependency on `factory-boy` — a test-data
library — whose `Trait` / `SubFactory` / `SelfAttribute` / `LazyAttribute`
machinery and stringly-typed `__`-path overrides are invisible to the type
checker and fail silently, we decided to replace the factory classes with
plain builder functions driven by one configuration object per toolchain
family and replaceable step builders, to achieve statically checked
composition, steps that cannot silently disagree on shared settings, one fewer
runtime dependency, and loud failures where the factories failed silently.

## Context

Every object these factories build — `Backend`, `OTFCompileWorkflow`,
`GTFNTranslationStep`, `DaCeTranslator`, the compilers — is already a frozen
dataclass. `factory-boy` added a second, parallel construction language on top:

- **Untyped.** The declarations are class attributes of a `Params` block, so
  `mypy` cannot check them. `src/` carried **8**
  `# type: ignore[assignment] # factory-boy typing not precise enough`
  suppressions solely to keep the factories quiet.

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
  `run_gtfn`. (The imperative backend was removed in #2877 while this change
  was pending, so `run_gtfn_imperative` is dropped rather than fixed; the
  incident remains the motivating evidence.)

- **A runtime dependency for a test-time concern.** `factory-boy` sat in
  `[project] dependencies`, shipped to every user, to compose four backends.

Whatever replaces the factories must still meet two concerns ADR 0017 lists
for toolchain configuration: some settings must be configured in **several
components in sync** (the target device reaches the translation step, the
compiler and the allocator), and users need to **switch out or tweak nested
steps** (a translation step without the GTIR transforms, a different build
system). `factory-boy` met both, untyped: `SelfAttribute("..device_type")` for
the first, `SubFactory` plus `__` paths for the second.

## Decision

Factory classes are replaced by **plain builder functions**; `factory-boy`
moves to the `test` dependency group, where the `cartesian` and `eve` IR
test-data factories keep using it for what it is designed for. The builders
follow three rules.

1. **Shared settings live in one configuration.** Each toolchain family has a
   frozen config dataclass, `GTFNConfig` and `DaCeConfig`, both extending
   `backend.ToolchainConfig`. It holds the settings that describe what the
   toolchain builds for — the device, the build type, the cache lifetime, the
   data layout, translation caching — and, for DaCe, the settings that couple
   the toolchain to its translation step (auto-optimize, the external
   workspace). Its defaults are read from `gt4py.next.config` when the config
   is created, which gives ADR 0017's precedence: an explicit argument wins
   over the user configuration, which wins over the builder default. Values
   derived from shared settings — the concrete device type from `gpu`, the
   allocator, the DaCe transient memory mode — are derived in exactly one
   place.

2. **Every step is created by a step builder that receives the config.** The
   step builders (`make_gtfn_translation`, `make_gtfn_bindings`,
   `make_gtfn_compiler`, `make_dace_translator`, …) take the config as their
   only positional argument and **step-local settings only** as keyword
   arguments. A shared setting therefore cannot be set for one step alone. A
   step is customized by passing a different builder to `make_*_toolchain` or
   `make_*_compile_workflow`: a `functools.partial` of the default builder to
   change a step-local setting, or any `Callable[[Config], Step]` to replace
   the step. Nested steps follow the same pattern: the GTFN build system is a
   step builder argument of `make_gtfn_compiler`.

3. **The toolchain builder owns the composition and checks custom steps.** It
   calls the step builders, then wraps the translation step in the cache, so a
   customization always lands on the bare step — the path that
   `run_gtfn_imperative` never reached. A step builder may be arbitrary user
   code that ignores the config, so the builder checks with
   `workflow.check_device_agreement` that a step recording a device agrees
   with the config. It checks, never mutates.

```python
gtfn.make_gtfn_toolchain(
    gtfn.GTFNConfig(gpu=True),
    name_postfix="_no_transforms",
    translation=functools.partial(gtfn.make_gtfn_translation, enable_itir_transforms=False),
)
```

`make_dace_backend` keeps its flat keyword signature as a front end over
`make_dace_toolchain`, so existing callers are unaffected.

## Consequences

- Composition is ordinary, statically checked Python. `mypy` rejects a
  misspelled step-local setting, a value of the wrong type, and an attempt to
  set a shared setting through a step builder
  (`partial(make_gtfn_translation, device_type=...)`); the same mistakes raise
  `TypeError` when the toolchain is built. The 8 factory-related
  `type: ignore` suppressions are gone. One remains, scoped and documented, in
  `make_gtfn_bindings`: `OTFCompileWorkflow` is not parameterized over the
  code spec, while `ExtensionGenerator` accepts only C++-like specs.
- Default and partially customized steps agree on the shared settings by
  construction, because they read them from the same config. A fully custom
  step builder can still ignore the config; the device check catches that
  case for the device only, and only for steps that record it as
  `device_type`. Invariants checked by the assembled pipeline itself, which
  would also cover `dataclasses.replace` on a built toolchain, are left to the
  pipeline rework.
- Step fields that must agree with other steps get no default, so a builder
  that forgets to pass one fails instead of silently using the default:
  `GTFNTranslationStep.device_type` no longer defaults to the CPU.
- A new shared setting is one config field, read where it is needed, instead
  of a keyword argument threaded through every builder layer.
- The price is two concepts instead of one (the config and the step
  builders); a `partial` is less discoverable than a keyword argument; the
  step builders re-list the step-local fields of their step; and the config
  must stay limited to shared settings or it turns into a grab-bag.
- Step builders run at build time and are not stored, so a `lambda` step
  builder does not make the toolchain unpicklable (offloading compilation to
  worker processes needs a picklable executor).
- One config means one default: a compile workflow built on its own now
  caches its translation step, like the toolchains always did.
  `GTFNConfig(cached_translation=False)` opts out.
- **`run_gtfn_no_transforms` is renamed** from `run_gtfn_cpu` to
  `run_gtfn_cpu_no_transforms`, removing the collision with `run_gtfn`. No
  cache is affected: the build cache keys on the entry-point name plus a
  fingerprint of the `ExtensionSource`, and the translation-cache directory
  is keyed on the literal backend family (`gtfn` / `dace`). `Backend.name`
  reaches only the metrics source key and one error message, so what the
  collision actually cost was two distinct backends sharing one metrics
  identity.
- All other pre-built toolchains are unchanged, verified field by field
  against the previous construction, and `make_dace_backend` builds
  field-identical toolchains for the same arguments.
- Downstream code migrates as `GTFNBackendFactory(gpu=on_gpu)` →
  `make_gtfn_toolchain(GTFNConfig(gpu=on_gpu))`, and
  `DaCeBackendFactory(..., otf_workflow__bare_translation__async_sdfg_call=False)`
  → `make_dace_backend(..., async_sdfg_call=False)`.

## Alternatives Considered

### Inject pre-built steps, used verbatim

A first version of this change had builders take shared settings as keyword
arguments and accept a pre-built step, used verbatim and checked for device
agreement. Changing one setting of an inner step then meant building the
whole step and repeating the shared settings in it:
`make_gtfn_backend(gpu=True, translation=GTFNTranslationStep(enable_itir_transforms=False))`
raised, because the injected step defaulted to the CPU, and the caller had to
re-derive the GPU device type (`CUPY_DEVICE_TYPE or CUDA`) the builder already
knew. Only the device was checked, so other shared settings could still
disagree silently, and each shared setting had to be threaded through every
builder layer by hand.

### Typed forwarding of per-step options

Builders could take the step-local settings of each inner step as a
`TypedDict` and forward them (`translation={"enable_itir_transforms": False}`).
That keeps `factory-boy`'s one-call ergonomics and is statically checked, and
leaving the shared settings out of the `TypedDict`s keeps them in sync. But
the `TypedDict`s mirror the step fields and drift from them, they cannot
replace a step (which needs a second, instance-injection mechanism with the
problems above), and the builders still forward every shared setting by hand
through each layer, where a forgotten forward silently falls back to the
step's default.

### Stamp shared settings onto injected steps

A `with_changes(step, **changes)` helper would stamp the shared settings onto
whichever step is present, applying only the fields the target declares.
Silently ignoring the fields a target does not declare reproduces exactly the
failure mode that motivated this ADR — the `run_gtfn_imperative` bug is what a
silent no-op looks like after a year. Checking is the same amount of
introspection with the opposite failure mode.

### Edit the built toolchain

`dataclasses.replace` on a built toolchain keeps working, but the caller must
know the nesting of wrappers (`executor.translation.step`) — the path the
`run_gtfn_imperative` override never reached — and the values the builder
derived (cache folders, name, allocator) are not recomputed.
