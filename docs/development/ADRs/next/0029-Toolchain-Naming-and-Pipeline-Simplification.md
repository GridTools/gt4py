---
tags: [backend, otf, toolchain, workflows, naming, observability]
---

# Toolchain Naming and Pipeline Simplification

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-07-30
- **Updated**: 2026-07-30

In the context of the on-the-fly compilation toolchain, facing a root object
whose names (`Backend`, `transforms`, `executor`) no longer match the
established *toolchain* vocabulary and a workflow-combinator framework whose
reading cost exceeds its value, we decided to rename the root object to
`Toolchain` (with `frontend` / `backend` halves), rename the envelope and
stage types accordingly, replace the combinator tower with explicit, fully
typed pipelines, and add a sanctioned stage-observability seam. We considered
keeping the combinators and trimming only the dead ones, and accept that the
renames break the old names outright, rotate the persistent translation-cache
keys (the second such rotation in this stack), and impose a migration burden
on downstream code that subclassed the combinators.

This partially supersedes [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md)
(the workflow-combinator framework and the `otf.step_types` naming — its stage
vocabulary and build-system decisions remain valid) and refines
[0017 - Toolchain Configuration](0017-Toolchain-Configuration.md) (whose term
*toolchain* becomes the name of the root object; its configuration decisions
remain valid).

This builds on [0028 - Plain Builders Instead of factory-boy Factories](0028-Plain-Builders-Instead-of-Factories.md),
which replaced the factory classes that constructed these objects, so the
renames below land in plain, type-checked builder code.

## Context

ADR 0017 already defines **toolchain** as "all the code components that work
together to go from DSL code to an optimized, runnable python callable" —
JIT/OTF pipelines, transformation passes, lowerings, parsers. The code grew
into exactly that shape while keeping older names: the root object is called
`Backend`, its frontend half `transforms`, its backend half `executor`, and
the pipeline pieces are spread over five vaguely-named modules (`workflow`,
`toolchain`, `definitions`, `stages`, `recipes`).

Separately, the workflow-combinator framework introduced by ADR 0011 had grown
to thirteen abstractions (`Workflow`, two mixins, `NamedStepSequence`,
`MultiWorkflow`, `StepSequence`, `CachedStep`, `SkippableStep`, `make_step`,
the `ConcreteArtifact` envelope and three adapters) to express function
composition (some of them already deleted as dead code in the preparatory
cleanup PRs). Measured against actual use, only `CachedStep` is a deep module;
the others are shallow wrappers around `Callable[[S], T]`, and the
`NamedStepSequence.__call__` reflection loop is `Any`-typed, defeating the
static typing ADR 0011 prized.

Finally, the seam between the toolchain and its consumers has no sanctioned
interface for observing or running intermediate stages: the one consumer that
needs an intermediate stage (the dace `__sdfg__` path) duck-types into the
default pipeline and mutates its stages.

## Decision

### Naming

| Old                          | New                               |
| ---------------------------- | --------------------------------- |
| `Backend`                    | `Toolchain`                       |
| `Backend.transforms`         | `Toolchain.frontend`              |
| `Backend.executor`           | `Toolchain.backend`               |
| `Backend.load_artifact()`    | `Toolchain.loading`               |
| `OTFCompileWorkflow`         | `CompilePipeline`                 |
| `ConcreteArtifact` / `.data` | `ProgramWithArgs` / `.definition` |
| `CompilableProgramDef`       | `CompilableProgram`               |

These are hard renames: no deprecation aliases are kept, so the old names stop
resolving in the same release. The public `gtx.*` names (`gtx.gtfn_cpu`,
`gtx.wait_for_compilation`, ...) are unchanged, with one exception —
`gtx.typing.Backend` becomes `gtx.typing.Toolchain`.

We considered shipping one-release read aliases for the renamed attributes and
re-export shims for the moved modules. We rejected that: an alias that silently
keeps working is exactly what lets a downstream stay on the old vocabulary past
the release where it was supposed to migrate, and the aliases would have been
partial anyway (constructing with the old keyword names could not be aliased
without hand-writing `__init__`, so `Backend(executor=...)` would have raised
`TypeError` regardless). A single loud break at a known release is easier to
act on than a half-working compatibility layer. `CompilableProgram` stays a type
alias (of the parameterized envelope) for now.

The last row is not a pure rename. `Backend.load_artifact()` was an overridable
method, and a toolchain that needed backend-specific runtime data in the loaded
program subclassed `Toolchain` to override it (see
[0027 - External Workspace Memory for DaCe Transients](0027-External_Workspace_Memory.md)).
`Toolchain.loading` makes that seam a field like the other three — a plain step
defaulting to `artifacts.load_artifact` — so customization stays
composition-time, consistent with the rest of this ADR. It also keeps
subclassing off the load path: per
[0024 - Compilation Runners](0024-Compilation-Runners.md), a toolchain that
customizes `compile` is opaque to the process-pool runner, and a
customization mechanism built on subclassing invites exactly that. The step
lives on `Toolchain` rather than inside `CompilePipeline` because loading also
happens off the pipeline — `CompiledProgramsPool` loads artifacts straight from
a compilation future — and because monolithic backends have no `CompilePipeline`
to hang it on.

### Pipeline, not combinators

Named pipelines become frozen dataclasses with an explicit, fully typed
`__call__`; steps are plain callables (`Step[S, T] = Callable[[S], T]`);
customization stays composition-time via `dataclasses.replace`. The
combinator framework (both mixins, `NamedStepSequence`, `MultiWorkflow`,
`StepSequence`, `make_step`, `.chain`, the three adapters) is deleted;
`CachedStep` is kept unchanged.

What ADR 0011's decisions become:

| ADR 0011 requirement                        | Kept by                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| Named steps, order visible                  | dataclass fields + explicit `__call__` (order literally visible)        |
| Statically typed composition                | explicit `__call__` — checked end-to-end, unlike the reflection loop    |
| Customization at composition, not via flags | `dataclasses.replace` on frozen pipelines                               |
| Steps compose across backends               | `Step[S, T]` is a `Callable` — every existing step already satisfies it |
| Linear workflows                            | unchanged (`Transforms` keeps its input-dependent step *selection*)     |

### Stage observability

A `stage_hook(name, artifact)` event hook (on the existing
`instrumentation.hook_machinery`) is emitted by `CompilePipeline` and
`Transforms` after each step; artifacts are treated opaquely.
`GT4PY_DUMP_STAGES=<dir>` registers a subscriber that writes each stage
artifact per program. `Toolchain.translate(definition, compile_time_args)` is
the sanctioned partial run (frontend + translation only); it narrows to the
standard `CompilePipeline` shape and raises a clear error on monolithic
backends. Per-call step options are deliberately not offered — a caller
needing a variant step builds a variant pipeline with `dataclasses.replace`.

### Phasing

The decisions in this ADR land as a stack of PRs. The naming decisions above
land first, except the `OTFCompileWorkflow` → `CompilePipeline` rename, which
lands with the pipeline PR; the "Pipeline, not combinators" and "Stage
observability" decisions are implemented in follow-up PRs of the same stack.

## Consequences

- The dace `__sdfg__` reach-in (duck-typing through `executor.translation`
  and mutating frozen stages) will be replaced by a dace-owned translate-only
  step built with `dataclasses.replace`.
- Downstream code subclassing the deleted combinators or overriding
  `Transforms.step_order` must migrate to explicit `__call__` composition.
- Downstream code must migrate to the new names in the same release: the old
  ones are removed, not aliased. That includes the modules dissolved along the
  way — `otf.definitions`, `otf.code_specs`, `otf.recipes` and `otf.toolchain`
  cease to exist, and `gtx.typing.Backend` becomes `gtx.typing.Toolchain`.
- ADR 0011's `otf.step_types` naming section no longer describes the code.
- Cache keys rotate: fingerprints embed the qualified class and field names,
  so the `ProgramWithArgs` / `definition` rename changes the keys of the
  persistent translation caches (gtfn and dace). This is the *second* of two
  rotations in this stack — moving the same class from `otf.toolchain` to
  `otf.workflow` already rotated them once, since the qualified name alone
  is enough. Both are subsumed by `BUILD_CACHE_VERSION_ID`, which rotates
  these keys at every release, so the effect is a cold rebuild rather than a
  stale hit. The two collapse into a single cold rebuild if the whole stack
  lands in one release, which is the concrete argument for not spreading it
  across releases.

## Alternatives considered

- **Keep the combinators but trim dead ones**: rejected — the tower is a
  shallow-module archetype and every recorded ADR 0011 requirement survives
  its removal; trimming keeps the reading cost and the `Any`-typed
  composition.
- **A per-call options API on `translate()`**: rejected — reconfiguration
  stays composition-time, ADR 0011's own rule.
