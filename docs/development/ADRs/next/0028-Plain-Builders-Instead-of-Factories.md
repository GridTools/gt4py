---
tags: [backend, otf, toolchain, workflows, dependencies]
---

# Plain Builders Instead of factory-boy Factories

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-08-20
- **Updated**: 2026-09-30

In the context of composing the GTFN and DaCe toolchains and their OTF compile
workflows, facing a production dependency on `factory-boy` — a test-data
library — whose `Trait` / `SubFactory` / `SelfAttribute` / `LazyAttribute`
machinery and stringly-typed `__`-path overrides are invisible to the type
checker and fail silently, we decided to replace the factory classes with
plain builder functions driven by one configuration object per toolchain
family and replaceable step builders, to achieve statically checked
composition, steps that agree on shared settings by construction, and one
fewer runtime dependency.

## Context

Every object the toolchain factories built was already a frozen dataclass.
`factory-boy` added a second, parallel construction language on top of them:

- **Untyped.** The declarations are class attributes of a `Params` block, so
  `mypy` cannot check them, and the factories needed `type: ignore`
  suppressions to type-check at all.
- **Silently wrong.** Overrides are `__`-delimited strings resolved at
  runtime. When a path does not resolve, nothing happens. This was not
  hypothetical: an override meant to switch a backend to imperative code
  generation (`otf_workflow__translation__use_imperative_backend=True`) never
  reached the translation step, because a caching trait had wrapped it, so the
  backend was the declarative one under another name for its whole life.
- **A runtime dependency for a test-time concern.** `factory-boy` was shipped
  to every user to compose a handful of backends.

Whatever replaces the factories must still meet two concerns ADR 0017 lists
for toolchain configuration: some settings must be configured in **several
components in sync** (the target device reaches the translation step, the
compiler and the allocator), and users need to **switch out or tweak nested
steps** (a translation step without the GTIR transforms, a different build
system). `factory-boy` met both, untyped: `SelfAttribute("..device_type")` for
the first, `SubFactory` plus `__` paths for the second.

## Decision

Factory classes are replaced by **plain builder functions**; `factory-boy`
moves to the test dependencies, where the IR test-data factories keep using it
for what it is designed for. The builders follow three rules.

1. **Shared settings live in one configuration.** Each toolchain family has a
   frozen config dataclass extending `backend.ToolchainConfig`. It holds the
   settings that describe what the toolchain builds for — the device, the
   build type, the cache lifetime, the data layout, translation caching — and
   the family-specific settings that several steps must agree on. Its defaults
   are read from `gt4py.next.config` when the config is created, which gives
   ADR 0017's precedence: an explicit argument wins over the user
   configuration, which wins over the builder default. Values derived from
   shared settings are derived in exactly one place.

2. **Every step is created by a step builder that receives the config.** A
   step builder takes the config as its only positional argument and
   **step-local settings only** as keyword arguments, so a shared setting
   cannot be set for one step alone. A step is customized by passing a
   different builder to the toolchain or compile-workflow builder: a
   `functools.partial` of the default builder to change a step-local setting,
   or any `Callable[[Config], Step]` to replace the step. Nested steps follow
   the same pattern.

3. **The toolchain builder owns the composition.** It calls the step builders
   and then applies the wrappers, such as the translation cache, so a
   customization always lands on the bare step. A fully custom step builder is
   responsible for configuring its step consistently with the config it
   receives; the builders do not validate what it returns.

```python
make_gtfn_toolchain(
    GTFNConfig(gpu=True),
    name_postfix="_no_transforms",
    translation=functools.partial(make_gtfn_translation, enable_itir_transforms=False),
)
```

The pre-existing flat-keyword `make_dace_backend` is kept as a deprecated
front end over the config-based builder, for existing callers.

## Consequences

- Composition is ordinary, statically checked Python: a misspelled step-local
  setting, a value of the wrong type, or an attempt to set a shared setting
  through a step builder is a type error, and a `TypeError` when the toolchain
  is built.
- Default and partially customized steps agree on the shared settings by
  construction, because they read them from the same config. Steps from fully
  custom step builders are not checked.
- A step field that must agree with other steps should have no default, so a
  builder that forgets to pass it fails instead of silently using the default.
  Fields read by a single step, such as the build type of the build system,
  may keep a default; a custom step builder that creates such a component
  must pass the config value itself.
- A new shared setting is one config field, read where it is needed, instead
  of a keyword argument threaded through every builder layer.
- Step builders run at build time and are not stored, so a `lambda` step
  builder does not make the toolchain unpicklable.
- There is one flat config per toolchain family. It is not composable: a
  toolchain assembled from sub-toolchains with different shared settings would
  need a different structure. Nothing needs that today.
- Configuration is pulled by each step builder from the config rather than
  pushed down by the parent, so a step builder's signature does not show which
  shared settings it reads, and a setting a step gains later silently keeps its
  step default until its builder reads it from the config.
- The construction logic is spread over many small step builders, which makes
  the consistency between steps harder to see and requires unit tests per
  builder.
- The price is also two concepts instead of one (the config and the step
  builders), a `partial` is less discoverable than a keyword argument, the step
  builders re-list the step-local fields of their step, and the config must
  stay limited to shared settings or it turns into a grab-bag.

## Alternatives Considered

### Typed forwarding of per-step options

Builders could take the step-local settings of each inner step as a
`TypedDict` and forward them (`translation={"enable_itir_transforms": False}`).
That keeps `factory-boy`'s one-call ergonomics, keeps the configuration flow
from parent to child visible, and is statically checked: `mypy` checks the
keys when a `TypedDict` is unpacked into the step constructor, so a stale key
is an error, and leaving the shared settings out of the `TypedDict`s keeps
them in sync. Like the step builders chosen here, a `TypedDict` only exposes
the step settings someone added to it.

It was not chosen because replacing a whole step needs a second mechanism next
to the options, typically injecting a pre-built step, which brings back the
problems of the next alternative; and because every shared setting has to be
forwarded by hand through each builder layer.

### Inject pre-built steps, used verbatim

Builders could take shared settings as keyword arguments and accept a
pre-built step, used verbatim. Changing one setting of an inner step then
means building the whole step and repeating the shared settings in it, which
the builder already knew; a GPU toolchain with a CPU translation step is only
caught if the builder checks for it.

### Stamp shared settings onto injected steps

A `with_changes(step, **changes)` helper would stamp the shared settings onto
whichever step is present, applying only the fields the target declares.
Silently ignoring the fields a target does not declare reproduces the failure
mode that motivated this ADR.

### Edit the built toolchain

`dataclasses.replace` on a built toolchain keeps working, but the caller must
know the nesting of wrappers — the path the silently dropped override above
never reached — and the values the builder derived (cache folders, name,
allocator) are not recomputed.
