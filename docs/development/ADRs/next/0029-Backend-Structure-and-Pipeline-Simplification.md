---
tags: [backend, otf, toolchain, workflows, naming, observability]
---

# Backend Structure and Pipeline Simplification

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-07-30
- **Updated**: 2026-10-29

In the context of the object users pass as `backend=` and the on-the-fly compilation route behind
it, facing a root object that mixes a user-facing concept (where and how a program runs) with
internal composition (which frontend lowers it, which steps compile it), and a workflow-combinator
framework whose reading cost exceeds its value, we decided to keep the established name `Backend`
for the user-facing object but make it an abstract base with a `name` and an `allocator`, with a
concrete `CompiledBackend` kind that carries a `compiler` and a `loader`; to take the frontend out
of the backend into a single global default composed with a compiled backend by an internal
`Toolchain`; to dispatch on the backend kind in one place; to rename the envelope and stage types;
to replace the combinator tower with explicit, fully typed pipelines; and to add a sanctioned
stage-observability seam. We accept that the changes break the old names outright (no aliases),
that a backend can no longer customize `compile` or its frontend, that the persistent
translation-cache keys rotate once more, and that downstream code that subclassed `Backend` or the
combinators must migrate.

This partially supersedes [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md) (the
workflow-combinator framework and the `otf.step_types` naming; its stage vocabulary and
build-system decisions remain valid), [0024 - Compilation Runners](0024-Compilation-Runners.md)
(the fallback for backends that customize `compile` no longer exists; the rest is unchanged),
[0027 - External Workspace Memory for DaCe Transients](0027-External_Workspace_Memory.md) (the
workspace is injected by a `CompiledBackend.loader`, not a `DaCeBackend` subclass) and
[0028 - Plain Builders Instead of factory-boy Factories](0028-Plain-Builders-Instead-of-Factories.md)
(builder and config names, and the deprecated flat-keyword `make_dace_backend`), and refines
[0017 - Toolchain Configuration](0017-Toolchain-Configuration.md) (its term *toolchain* names the
internal composition of the frontend with a compiled backend; its configuration decisions remain
valid). Embedded execution as `Backend` objects is decided separately, in
[ADR 0030](0030-Embedded-Backends.md) (follow-up).

## Context

ADR 0017 defines **toolchain** as "all the code components that work together to go from DSL code
to an optimized, runnable python callable". The root object users pass as `backend=` had grown
into a toolchain in that sense: `Backend(name, executor, allocator, transforms)` held the frontend
transforms, the compile workflow, the loading logic (an overridable `load_artifact()` method) and
the allocator. Users, however, regard **backend** as the established name for "where and how my
program runs", covering both halves; renaming the user-facing object to `Toolchain` (the first
version of this ADR) was rejected in review for that reason, and because a complete rename would
have reached every `backend=` keyword, `with_backend`, `Program.backend` and the test framework.

Looking at what the fields are actually used for:

- **The frontend is not a per-backend choice.** Every in-tree backend uses the same
  `DEFAULT_TRANSFORMS`; the one that built its own (`roundtrip.gtir`, a leftover of #1569) built an
  identical one, differing only in owning separate in-memory caches. `Program` and `FieldOperator`
  lower with `DEFAULT_TRANSFORMS` whenever they have no backend, and `FieldOperator.__gt_itir__`
  ignores the backend's frontend on purpose (an operator called from a program must lower the same
  way regardless of the operator's own backend). A per-backend frontend was a configuration point
  that was not honoured consistently.
- **The allocator is the one thing every kind of execution needs.** Embedded execution has no
  frontend, compiler or loader, but it allocates fields (tests and applications allocate with
  `allocator=backend`); it is represented today by `backend=None`, which has no allocator, no name
  and cannot be distinguished from "not configured".
- **`compile` and `load_artifact` were overridable methods.** A backend needing runtime data in the
  loaded program subclassed the root object (ADR 0027's `DaCeBackend`), and ADR 0024 had to add an
  "opaque" fallback for backends overriding `compile`, which cannot be decomposed for the process
  pool. Test doubles relied on duck-typing `compile`.
- **`name` had no stated meaning.** It is used as a label in metrics, logs and diagnostics, while
  builder docstrings claimed it "must stay unique" without anything enforcing or relying on it.

Separately, the workflow-combinator framework introduced by ADR 0011 had grown to thirteen
abstractions (`Workflow`, two mixins, `NamedStepSequence`, `MultiWorkflow`, `StepSequence`,
`CachedStep`, `SkippableStep`, `make_step`, the `ConcreteArtifact` envelope and three adapters) to
express function composition (some of them already deleted as dead code in the preparatory
cleanup PRs). Measured against actual use, only `CachedStep` is a deep module; the others are
shallow wrappers around `Callable[[S], T]`, and the `NamedStepSequence.__call__` reflection loop is
`Any`-typed, defeating the static typing ADR 0011 prized.

Finally, the seam between the compile route and its consumers has no sanctioned interface for
observing or running intermediate stages: the one consumer that needs an intermediate stage (the
dace `__sdfg__` path) duck-types into the default pipeline and mutates its stages.

## Decision

### Backend structure

Sketch (step types as of the pipeline PR; until then `compiler` and `frontend` are typed
`workflow.Workflow[...]`):

```python
@dataclasses.dataclass(frozen=True)
class Backend:  # abstract: __post_init__ rejects `type(self) is Backend`
    name: str
    allocator: Allocator  # array namespace or field-buffer allocator (factory)


@dataclasses.dataclass(frozen=True)
class CompiledBackend(Backend):
    allocator: FieldBufferAllocatorProtocol
    compiler: Step[CompilableProgram, CompilationArtifact]
    loader: Callable[[CompilationArtifact], ExecutableProgram] = artifacts.load_artifact


@dataclasses.dataclass(frozen=True)
class Toolchain:  # internal
    backend: CompiledBackend
    frontend: Step[ConcreteProgramDef, CompilableProgram] = DEFAULT_TRANSFORMS
```

- **`Backend` stays the user-facing name** (`gtx.typing.Backend`, `backend=`, `with_backend`,
  `Program.backend`). It is an abstract base with two fields. Abstractness is enforced in
  `__post_init__` rather than with abstract methods: there are none, and a required dataclass field
  cannot implement an inherited abstract property.
- **`CompiledBackend`** is the kind that compiles. `compiler` turns the lowered program into a
  loadable artifact and may run in a worker process; `loader` turns the artifact into a callable
  and always runs in the calling process. Both are plain fields: customization is
  composition-time (`dataclasses.replace(backend, compiler=...)`), never subclassing. `compile` is
  not a method of any `Backend`.
- **Embedded execution gets its own kind**, `EmbeddedBackend(Backend)`, a sibling of
  `CompiledBackend` rather than a mode of it. Introducing it, together with retiring
  `backend=None`, is decided in [ADR 0030](0030-Embedded-Backends.md) and lands separately; until
  then embedded execution is still `backend=None`.
- **The frontend is global.** `DEFAULT_TRANSFORMS` is the only frontend; it is not a field of any
  `Backend`. The internal `Toolchain` composes it with a `CompiledBackend` where a compilation
  happens (the compiled-programs pool, one-shot compiles, partial runs). `Toolchain` is not
  exported from `gt4py.next` or `gtx.typing`; its `frontend` field exists for gt4py's own tests
  and future internal use, not as a user customization point.
- **Dispatch on the backend kind lives in one helper** used by `Program`, `FieldOperator` and scan
  operators: a `CompiledBackend` goes through the compiled-programs pool, embedded execution runs
  the definition, anything else raises `TypeError`. Construction (`@program`, `with_backend`,
  `dataclasses.replace`) additionally rejects objects that are not `Backend`s at all.
- **No opaque compile path.** With `compile` gone from the user-facing object, the
  compilation-task layer always decomposes a compile into frontend (main process), compiler
  (possibly offloaded) and loader (main process). A `compiler` that cannot be pickled still falls
  back to the calling thread, as ADR 0024 describes.
- **`name` is a label.** It identifies the backend in metrics, logs and diagnostics. It is expected
  to be unique among the backends used in a process; backends sharing a name are reported under the
  same label. gt4py does not use it to identify a backend: not as a cache key and not in
  fingerprints (`Program.backend` is excluded from fingerprints; translation caches are keyed on
  the program and the step). Uniqueness is documented, not enforced. Generated builder names encode
  only the device (and for DaCe the optimization level); callers distinguish other variants with
  `name_postfix`. Code must not infer a backend's kind from its name: the dace SDFG conversion
  checks the type of the translation step inside `compiler` instead, and other code should do the
  same (`isinstance`, `is_dace_backend`).

### Loading is a field

`Backend.load_artifact()` was an overridable method, and a backend that needed backend-specific
runtime data in the loaded program subclassed the root object to override it (ADR 0027).
`CompiledBackend.loader` makes that seam a field — a plain callable defaulting to
`artifacts.load_artifact` — so customization stays composition-time, consistent with the rest of
this ADR, and the `DaCeBackend` subclass is replaced by a `DaCeLoader` holding the external
workspace (`backend.loader.external_workspace`). The loader lives on the backend rather than
inside the compile pipeline because loading also happens off the pipeline — the compiled-programs
pool loads artifacts straight from a compilation future — and because monolithic compilers have no
pipeline to hang it on.

### Naming

| Old (`main`)                                   | New                                         |
| ---------------------------------------------- | ------------------------------------------- |
| `Backend` (concrete)                           | `Backend` (abstract), `CompiledBackend`     |
| `Backend.executor`                             | `CompiledBackend.compiler`                  |
| `Backend.transforms`                           | removed; internal `Toolchain.frontend`      |
| `Backend.load_artifact()`                      | `CompiledBackend.loader`                    |
| `Backend.compile()`                            | internal `Toolchain.compile()`              |
| `DaCeBackend` / `.external_workspace`          | `DaCeLoader` / `.loader.external_workspace` |
| `make_gtfn_toolchain` / `make_dace_toolchain`  | `make_gtfn_backend` / `make_dace_backend`   |
| `make_dace_backend(gpu=..., ...)` (deprecated) | removed (its name is reused)                |
| `ToolchainConfig`                              | `CompiledBackendConfig`                     |
| `OTFCompileWorkflow`                           | `CompilePipeline`                           |
| `ConcreteArtifact` / `.data`                   | `ProgramWithArgs` / `.definition`           |
| `CompilableProgramDef`                         | `CompilableProgram`                         |
| `ItirShim.definition`                          | `ItirShim.operator_def`                     |

The builders and their config base follow the new vocabulary, since they build a `CompiledBackend`.
The flat-keyword `make_dace_backend` was deprecated one release earlier (ADR 0028); it is removed
rather than kept because the config-based builder takes its name, and two builders with one name
cannot coexist. A call in the removed shape fails with a `TypeError` that explains the new
signature.

These are hard renames: no deprecation aliases are kept, so the old names stop resolving in the
same release. The public `gtx.*` names (`gtx.gtfn_cpu`, `gtx.wait_for_compilation`,
`gtx.typing.Backend`, ...) are unchanged.

We considered shipping one-release read aliases for the renamed attributes and re-export shims for
the moved modules. We rejected that: an alias that silently keeps working is exactly what lets a
downstream stay on the old vocabulary past the release where it was supposed to migrate, and the
aliases would have been partial anyway (constructing with the old keyword names could not be
aliased without hand-writing `__init__`, so `Backend(executor=...)` would have raised `TypeError`
regardless). A single loud break at a known release is easier to act on than a half-working
compatibility layer. `CompilableProgram` stays a type alias (of the parameterized envelope) for
now.

### Pipeline, not combinators

Named pipelines become frozen dataclasses with an explicit, fully typed `__call__`; steps are
plain callables (`Step[S, T] = Callable[[S], T]`); customization stays composition-time via
`dataclasses.replace`. The combinator framework (both mixins, `NamedStepSequence`,
`MultiWorkflow`, `StepSequence`, `make_step`, `.chain`, the three adapters) is deleted;
`CachedStep` is kept, minus the mixin-provided `.replace` / `.chain`. The standard three-step `compiler` of the compiled backends
becomes `CompilePipeline` (translation, bindings, compilation); the frontend `Transforms` keeps its
input-dependent step selection.

What ADR 0011's decisions become:

| ADR 0011 requirement                        | Kept by                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| Named steps, order visible                  | dataclass fields + explicit `__call__` (order literally visible)        |
| Statically typed composition                | explicit `__call__` — checked end-to-end, unlike the reflection loop    |
| Customization at composition, not via flags | `dataclasses.replace` on frozen pipelines and backends                  |
| Steps compose across backends               | `Step[S, T]` is a `Callable` — every existing step already satisfies it |
| Linear workflows                            | unchanged (`Transforms` keeps its input-dependent step *selection*)     |

Static checking covers the stage types a step consumes and produces; it does not yet cover which
language their code is in. The `CompilePipeline` fields use unparameterized `ProgramSource` /
`ExtensionSource`, so mypy accepts a pipeline that mixes C++ and SDFG steps. Parameterizing
`CompilePipeline` over the code spec only helps once the builders return parameterized pipelines,
so it is left to a follow-up.

### Cross-step agreement is checked on construction

The builders of [0028](0028-Plain-Builders-Instead-of-Factories.md) keep default and partially
customized steps in sync, but leave fully custom step builders to the caller, and
`dataclasses.replace` lets a pipeline or a backend be assembled without any builder. The objects
that assemble steps therefore check device agreement themselves, in `__post_init__`.
`dataclasses.replace` re-runs that, so the check covers builders, direct construction and variants
alike:

- `CompilePipeline`: all steps that declare a device (the `backend.DeviceConfigurable` protocol,
  looking through `CachedStep`) declare the same one, which is exposed as
  `CompilePipeline.device_type`.
- `CompiledBackend`: the allocator's `__gt_device_type__` matches the device the `compiler`
  declares.

These checks never mutate. They only see components that declare a device: a custom step without
`device_type` is not checked. Settings shared by the steps of one backend family, such as the DaCe
external workspace and its `transient_memory_mode`, are that family's responsibility and are not
checked here.

### Stage observability

A `stage_hook(name, artifact)` event hook (on the existing `instrumentation.hook_machinery`) is
emitted by `CompilePipeline` and `Transforms` after each step; artifacts are treated opaquely.
`GT4PY_DUMP_STAGES=<dir>` registers a subscriber that writes each stage artifact per program.
`Toolchain.translate(definition, compile_time_args)` is the sanctioned partial run (frontend +
translation only) for gt4py's own consumers, such as the dace SDFG conversion; it narrows to the
standard `CompilePipeline` shape of `backend.compiler` and raises a clear error on monolithic
compilers. Per-call step options are deliberately not offered — a caller needing a variant step
builds a variant backend with `dataclasses.replace(backend, compiler=dataclasses.replace(...))`.

### Phasing

The decisions in this ADR land as a stack of PRs. The backend structure, loader, naming (except
the `OTFCompileWorkflow` → `CompilePipeline` rename) and the removal of the opaque compile path
land first (#2741); `EmbeddedBackend` and the retirement of `backend=None`, decided separately in
[ADR 0030](0030-Embedded-Backends.md), follow (#2956); then stage observability (#2742); the
pipeline simplification, together with the deferred `CompilePipeline` rename and the
construction-time device checks, lands last (#2743).

## Consequences

- `gtx.typing.Backend`, `backend=`, `with_backend` and `Program.backend` keep their names; code
  that only passes backend instances around (`gtx.gtfn_cpu`, `run_dace_cpu`, ...) is unaffected.
- Code that constructs or introspects backends must migrate: `CompiledBackend(name=..., allocator=..., compiler=..., loader=...)`, `.compiler` instead of `.executor`, `.loader`
  instead of `load_artifact()`, `make_*_backend` instead of `make_*_toolchain` and
  `CompiledBackendConfig` instead of `ToolchainConfig`. The old names are removed, not aliased; that includes `Backend` being instantiable (it is now abstract) and the
  dissolved modules of the pipeline PR (`otf.definitions`, `otf.code_specs`, `otf.recipes`,
  `otf.toolchain`).
- A backend can no longer change the frontend or override `compile`/`load_artifact`. Behaviour is
  customized by replacing `compiler` or `loader`; duck-typed backend objects are rejected when the
  program is constructed.
- Every compile is decomposable, so the process-pool runner never needs an opaque path; the only
  remaining reason to compile in the calling thread is an unpicklable `compiler`.
- The dace `__sdfg__` reach-in (duck-typing through the compile pipeline's `translation` and
  mutating frozen stages) is replaced by a dace-owned translate-only variant built with
  `dataclasses.replace`, used through `Toolchain.translate`.
- Downstream code subclassing the deleted combinators or overriding `Transforms.step_order` must
  migrate to explicit `__call__` composition.
- ADR 0011's `otf.step_types` naming section no longer describes the code.
- Cache keys rotate: fingerprints embed the qualified class and field names, so the
  `ProgramWithArgs` / `definition` rename changes the keys of the persistent translation caches
  (gtfn and dace). This is the *second* of two rotations in this stack — moving the same class from
  `otf.toolchain` to `otf.workflow` already rotated them once, since the qualified name alone is
  enough. What makes each a cold rebuild rather than a stale hit is that the key itself changes, so
  entries written under the old name are never looked up; `BUILD_CACHE_VERSION_ID` (the gt4py
  version) is a separate salt on top, which means the rotation costs nothing extra to anyone who
  crosses a release — or, for a source install, any commit — boundary anyway. The two collapse into
  a single cold rebuild if the whole stack lands in one release, which is the concrete argument for
  not spreading it across releases. The backend restructuring itself rotates nothing:
  `Program.backend` is excluded from fingerprints and no fingerprinted class gains or loses a field.

## Alternatives considered

- **Rename the user-facing `Backend` to `Toolchain`** (the first version of this ADR): rejected in
  review. "Backend" is how users name the choice of execution, frontend included; the rename would
  have had to reach every `backend=` keyword, `with_backend`, `Program.backend` and the test
  framework to be consistent, and a partial rename produced code like `self.backend.backend`.
- **One concrete `Backend` with an optional `compiler`** (embedded = no compiler): rejected; every
  consumer would branch on `compiler is None`, the type would not tell the two kinds apart, and an
  embedded-specific behaviour (e.g. a future `jax.jit` execution) would have nowhere to go.
- **Keep the frontend as a field of `CompiledBackend`**: rejected; it was not honoured
  consistently (see Context), embedded execution has none, and no backend uses a different one.
- **`allocator` as an abstract property on `Backend`**: does not work with dataclasses — a
  required field in a subclass does not implement an inherited abstract property, and `dataclass`
  re-runs `abc.update_abstractmethods`, so the subclass stays abstract.
- **Enforce name uniqueness** (registry or check at construction): rejected; nothing in gt4py relies
  on uniqueness, test doubles and `dataclasses.replace` variants legitimately share names, and a
  global registry would be process state for a label.
- **Keep overridable `compile` with the opaque process-pool fallback**: rejected; it made the
  process pool's behaviour depend on how a backend was written, and every known override (test
  doubles) is expressible by wrapping `compiler`.
- **Dispatch by polymorphism** (a method on `Backend`, e.g. `backend.run(program, args)` or
  `backend.make_pool(...)`, instead of one `isinstance` helper): rejected; what happens per kind
  (lowering with the global frontend, the compiled-programs pool, the embedded context and its
  hooks) belongs to the `ffront` program objects, which import `backend.py` — moving it onto
  `Backend` would invert that dependency, and an overridable method would reopen the "customized
  `compile`" problem this ADR removes. The set of kinds is closed and small.
- **Keep `make_*_toolchain` / `ToolchainConfig`**: rejected; public builders named after an
  internal class but returning a `CompiledBackend` would keep the vocabulary split this ADR
  removes.
- **Keep the combinators but trim dead ones**: rejected — the tower is a shallow-module archetype
  and every recorded ADR 0011 requirement survives its removal; trimming keeps the reading cost and
  the `Any`-typed composition.
- **A per-call options API on `translate()`**: rejected — reconfiguration stays composition-time,
  ADR 0011's own rule.
