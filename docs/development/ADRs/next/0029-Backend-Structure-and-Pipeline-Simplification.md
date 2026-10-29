---
tags: [backend, otf, toolchain, workflows, naming, observability]
---

# Backend Structure and Pipeline Simplification

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-07-30
- **Updated**: 2026-10-29

In the context of the compilation route behind `backend=`, we decided to keep a public,
abstract `Backend` describing execution and allocation, with `CompiledBackend` carrying a compiler
and loader, and an internal `Toolchain` composing it with the global frontend. We use explicit,
fully typed pipelines and a sanctioned interface for observing and running intermediate stages.
We accept hard breaks in internal names, removal of backend-specific frontends and overridable
compilation/loading methods, a persistent translation-cache rotation, and downstream subclass migration.

## Decision

### Concepts and composition

A **toolchain** is the complete route from DSL code to an optimized, runnable Python callable.

| Concept           | Responsibility and interface                                                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Backend`         | Public abstract execution choice: a `name` and an `allocator`. Used by `gtx.typing.Backend`, `backend=`, `with_backend` and `Program.backend`.                     |
| Allocator         | Provides field buffers; the base accepts an array namespace or a field-buffer allocator (factory), while compiled backends require `FieldBufferAllocatorProtocol`. |
| `CompiledBackend` | The backend kind that compiles: owns `compiler`, `loader` and the allocator for the target device.                                                                 |
| Frontend          | Lowers a program definition and its arguments into a `CompilableProgram`.                                                                                          |
| Global frontend   | `DEFAULT_FRONTEND`, shared by all backends; no backend owns or selects a frontend.                                                                                 |
| Compiler          | Turns a `CompilableProgram` into a loadable `CompilationArtifact`, possibly in a worker process.                                                                   |
| Loader            | Turns that artifact into an `ExecutableProgram` in the calling process; defaults to `artifacts.load_artifact`.                                                     |
| `Toolchain`       | Internal composition of the global frontend with a `CompiledBackend`, created where compilation happens.                                                           |

The objects are frozen dataclasses. This sketch uses the target pipeline types; in #2741,
`compiler` and `frontend` still use `workflow.Workflow[...]`:

```python
@dataclasses.dataclass(frozen=True)
class Backend:  # __post_init__ rejects type(self) is Backend
    name: str
    allocator: Allocator


@dataclasses.dataclass(frozen=True)
class CompiledBackend(Backend):
    allocator: FieldBufferAllocatorProtocol
    compiler: Step[CompilableProgram, CompilationArtifact]
    loader: Callable[[CompilationArtifact], ExecutableProgram] = artifacts.load_artifact


@dataclasses.dataclass(frozen=True)
class Toolchain:  # internal
    backend: CompiledBackend
    frontend: Step[ConcreteProgramDef, CompilableProgram] = DEFAULT_FRONTEND
```

`Backend` has no abstract methods; `__post_init__` enforces abstractness. `Toolchain` is not exported from
`gt4py.next` or `gtx.typing`; its frontend field is for GT4Py tests and future internal use, not
user customization. Embedded execution remains `backend=None` in this PR. A sibling
`EmbeddedBackend(Backend)`, rather than a mode of `CompiledBackend`, and retirement of `None`
are decided separately in [ADR 0030](0030-Embedded-Backends.md) (follow-up #2956).

### From definition to callable

```text
program definition + compile-time arguments
    -> global frontend                 [calling process]
    -> CompilableProgram
    -> backend.compiler                [runner: possibly a worker process]
    -> CompilationArtifact
    -> backend.loader                  [calling process]
    -> ExecutableProgram               [callable]
```

One helper dispatches on backend kind for `Program`, `FieldOperator` and scan operators:
a `CompiledBackend` uses the compiled-programs pool; `None` runs the definition in embedded
execution; unsupported kinds raise `TypeError`. Construction (`@program`, `with_backend`,
`dataclasses.replace`) rejects objects that are neither a `Backend` nor `None`.

The pool uses an internal `Toolchain`. The compilation-task layer runs its frontend main-side,
then submits the compiler and lowered program to the compilation runner. The process runner
returns an artifact through a future; the pool loads it on first use in the calling process.
A compiler that cannot be pickled falls back to the calling thread, with a warning. Serial and
thread runners remain available as specified in ADR 0024. Every task is decomposable; there is
no opaque compilation path. For one-shot compilation, internal `Toolchain.compile()` runs
frontend, compiler and loader synchronously in the calling thread. No `Backend` has a `compile`
method.

The loader belongs to the backend, outside its compiler: the pool loads directly from compilation
futures, and a monolithic compiler has no pipeline on which to place loading. DaCe supplies a
`DaCeLoader` carrying the caller's external workspace at `backend.loader.external_workspace` and
injecting it into the loaded program.

### Customization and names

Customize by composition, not subclassing: replace `compiler` or `loader` with
`dataclasses.replace(backend, compiler=...)`, or replace steps inside a pipeline.
`make_gtfn_backend` and `make_dace_backend` build compiled
backends, with family configs based on `CompiledBackendConfig`; ADR 0028's composition rules remain.

`name` is a label for metrics, logs and diagnostics. Uniqueness among backends in a process is
expected and documented, not enforced; shared names are reported under the same label. Names
identify neither cache entries nor fingerprints: `Program.backend` is excluded from fingerprints,
and translation caches key on the program and step. Generated names encode the device and, for
DaCe, whether auto-optimization is enabled; callers distinguish other variants with `name_postfix`.
Determine backend kind by types, never names (`isinstance`, `is_dace_backend`). The DaCe check
inspects the standard compiler's translation step, through a `CachedStep` if present.

### Explicit typed pipelines

In the target design, named pipelines are frozen dataclasses with explicit, fully typed `__call__`
methods and callable steps (`Step[S, T] = Callable[[S], T]`). The standard compiler becomes
`CompilePipeline`: translation, bindings, compilation.

| ADR 0011 requirement                        | Kept by                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| Named steps, order visible                  | dataclass fields + explicit `__call__` (order literally visible)        |
| Statically typed composition                | explicit `__call__` — checked end-to-end, unlike the reflection loop    |
| Customization at composition, not via flags | `dataclasses.replace` on frozen pipelines and backends                  |
| Steps compose across backends               | `Step[S, T]` is a `Callable` — every existing step already satisfies it |
| Linear workflows                            | unchanged (`Transforms` keeps its input-dependent step *selection*)     |

Static checking covers the stage types a step consumes and produces, not the language of their
code: the `CompilePipeline` fields use unparameterized `ProgramSource` / `ExtensionSource`, so mypy
accepts a pipeline mixing C++ and SDFG steps. Parameterizing `CompilePipeline` over the code spec
only helps once the builders return parameterized pipelines, so it is left to a follow-up.

### Cross-step agreement is checked on construction

The builders of [0028](0028-Plain-Builders-Instead-of-Factories.md) keep default and partially
customized steps in sync, but fully custom steps and `dataclasses.replace` can assemble a pipeline
or a backend without any builder. The assembling objects therefore check device agreement
themselves, in `__post_init__`, which `dataclasses.replace` re-runs:

- `CompilePipeline`: all steps that declare a device (the `backend.DeviceConfigurable` protocol,
  looking through `CachedStep`) declare the same one, exposed as `CompilePipeline.device_type`.
- `CompiledBackend`: the allocator's `__gt_device_type__` matches the device the `compiler`
  declares.

The checks never mutate and only see components that declare a device; a custom step without
`device_type` is not checked. Settings shared by the steps of one backend family, such as the DaCe
external workspace and its `transient_memory_mode`, are that family's responsibility.

### Stage observability

In the target design, a `stage_hook(name, artifact)` event hook on `instrumentation.hook_machinery` is
emitted by `CompilePipeline` and `Transforms` after each step; artifacts are treated opaquely.
`GT4PY_DUMP_STAGES=<dir>` registers a subscriber that writes each stage artifact per program.
`Toolchain.translate(definition, compile_time_args)` is the sanctioned partial run (frontend +
translation only) for gt4py's own consumers, such as the dace SDFG conversion; it narrows to the
standard `CompilePipeline` shape of `backend.compiler` and raises a clear error on monolithic
compilers. Per-call step options are deliberately not offered — a caller needing a variant step
builds a variant backend with `dataclasses.replace(backend, compiler=dataclasses.replace(...))`.
In #2742, DaCe's `__sdfg__` reach-in becomes a DaCe-owned translate-only variant built this way
and run through `Toolchain.translate`.

## Context and transition

The previous `Backend(name, executor, allocator, transforms)` combined frontend transforms,
compilation, overridable loading and allocation. Users call the execution choice a backend,
including the frontend, but its internal composition is a toolchain in ADR 0017's sense.

- Every backend used the default `Transforms`, except `roundtrip.gtir` (a #1569 leftover), which
  built identical transforms with separate in-memory caches. `Program` and `FieldOperator` used
  the default without a backend; `FieldOperator.__gt_itir__` ignored its backend's frontend to
  lower consistently inside programs. Per-backend frontend settings were inconsistently honoured.
- All execution kinds allocate: tests and applications use `allocator=backend`. Embedded execution
  needs no frontend, compiler or loader, but its `None` representation has neither allocator nor
  name and is indistinguishable from “not configured”.
- Overridable `compile` and `load_artifact` led to ADR 0027's `DaCeBackend` subclass and ADR 0024's
  opaque fallback for compilation that cannot be decomposed for the process pool.
  Test doubles duck-typed `compile`.
- Builder docstrings said names “must stay unique”, though nothing enforced or relied on it.
- ADR 0011's framework grew to thirteen abstractions: `Workflow`, two mixins,
  `NamedStepSequence`, `MultiWorkflow`, `StepSequence`, `CachedStep`, `SkippableStep`, `make_step`,
  `ConcreteArtifact` and three adapters. Some were already removed as dead code in preparatory
  PRs. Only `CachedStep` is a deep module; the rest wrap `Callable[[S], T]`, and the
  `NamedStepSequence.__call__` reflection loop is `Any`-typed, defeating static composition.
- Intermediate stages had no sanctioned interface. DaCe's `__sdfg__` historically mutated frozen
  stages; #2741 copies the cached frontend stage with `dataclasses.replace`, but still reaches
  into the translation step until #2742.

### Names and removed mechanisms

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
| `OTFCompileWorkflow`                           | `CompilePipeline` (in #2743)                |
| `ConcreteArtifact` / `.data`                   | `ProgramWithArgs` / `.definition`           |
| `CompilableProgramDef`                         | `CompilableProgram`                         |
| `ItirShim.definition`                          | `ItirShim.operator_def`                     |
| `DEFAULT_TRANSFORMS`                           | `DEFAULT_FRONTEND`                          |

The builders and config base are named for the `CompiledBackend` they build. The flat-keyword
`make_dace_backend`, deprecated one release earlier (ADR 0028), is removed because the config-based
builder takes its name: two builders cannot share it. Old-style calls raise `TypeError` explaining
the new signature.

These are hard renames: no deprecation aliases are kept, so the old names stop resolving in the
same release. The public `gtx.*` names (`gtx.gtfn_cpu`, `gtx.wait_for_compilation`,
`gtx.typing.Backend`, ...) are unchanged.

One-release attribute aliases and module re-export shims were rejected: they let downstream code
postpone migration and provide only partial compatibility. Old constructor keywords still require
a handwritten `__init__`; otherwise `Backend(executor=...)` raises `TypeError`. One explicit break
at a known release is easier to act on. `CompilableProgram` remains a parameterized-envelope type alias.

The overridable `compile`/`load_artifact` methods, `DaCeBackend` subclass, opaque “customized
compile” path and `CompilationTask.no_offload_reason` are removed. The pipeline PR removes the
combinator framework (both mixins, `NamedStepSequence`, `MultiWorkflow`, `StepSequence`,
`make_step`, `.chain` and the three adapters), and dissolves `otf.definitions`, `otf.code_specs`,
`otf.recipes` and `otf.toolchain`. `CachedStep` is kept, minus the mixin-provided `.replace` /
`.chain`.

### Cache-key rotation

Fingerprints include qualified class and field names. Renaming `ConcreteArtifact.data` to
`ProgramWithArgs.definition` changes persistent translation-cache keys for GTFN and DaCe. This
is the second rotation in the stack: moving the envelope from `otf.toolchain` to `otf.workflow`
already changed its qualified name. Old entries are never looked up under new keys, so each
rotation causes a cold rebuild, not a stale hit.

`BUILD_CACHE_VERSION_ID` is a separate salt, defaulting to the GT4Py version (and overridable via
`GT4PY_BUILD_CACHE_VERSION_ID`). When a release or source-install version change already changes
that salt, the rename adds no further rebuild. A source commit does not unconditionally change
the salt: version discovery can fall back to installed metadata, and an explicit override can
hold it constant. Users upgrading across the entire stack in one release see one cold rebuild;
this argues against spreading it across releases. Backend restructuring itself rotates nothing:
`Program.backend` is excluded and no fingerprinted class gains or loses a field from that change.

### Phasing and earlier decisions

#2741 lands the backend structure, loader, naming except `CompilePipeline`, and removal of the
opaque compile path. Embedded backends follow, decided separately in
[ADR 0030](0030-Embedded-Backends.md) (#2956). #2742 adds stage observability, including
`Toolchain.translate`; #2743 replaces the combinators, renames `OTFCompileWorkflow` to
`CompilePipeline` and adds the construction-time device checks. Thus the pipeline, agreement and
observability sections describe the target design, not APIs already available at #2741.

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
valid).

## Consequences

- Code only passing backend instances (`gtx.gtfn_cpu`, `run_dace_cpu`, ...) is unaffected.
- Construction and introspection must use `CompiledBackend(name=..., allocator=..., compiler=..., loader=...)` and the new names; `Backend` itself is no longer instantiable.
- Custom backend behavior moves to compiler/loader fields. Duck-typed backends fail at construction.
- Compile tasks are decomposable; process-mode fallback is limited to unpicklable compilers.
  Explicit serial execution remains supported.
- Subclasses of deleted combinators and overrides of `Transforms.step_order` must migrate to
  explicit `__call__` composition.
- Persistent translation caches rebuild under the changed keys, as described above.

## Alternatives considered

- **Rename public `Backend` to `Toolchain`** (this ADR's first version): rejected in review.
  Users call execution, including the frontend, a backend. Consistency would require renaming
  every `backend=`, `with_backend`, `Program.backend` and the test framework; a partial rename
  produced `self.backend.backend`.
- **One concrete `Backend` with optional `compiler`**: consumers must branch on `compiler is None`,
  types cannot distinguish execution kinds, and embedded-specific behavior (e.g. future `jax.jit`
  execution) has nowhere to go.
- **Frontend on `CompiledBackend`**: inconsistently honoured, absent in embedded execution,
  and identical across backends.
- **Abstract `allocator` property**: a required dataclass field cannot implement an inherited
  abstract property; `dataclass` reruns `abc.update_abstractmethods`, leaving subclasses abstract.
- **Enforce unique names** (registry or construction check): nothing relies on uniqueness;
  test doubles and `dataclasses.replace` variants legitimately share names. A global registry
  adds process state for a label.
- **Overridable `compile` with opaque fallback**: process-pool behavior depends on how a backend
  is written; all known overrides (test doubles) can wrap `compiler` instead.
- **Polymorphic dispatch** (`Backend.run(program, args)` or `Backend.make_pool(...)`): global
  lowering, the compiled-programs pool, embedded context and hooks belong to `ffront` program
  objects, which import `backend.py`. Moving them onto `Backend` inverts that dependency;
  overridable methods reopen the customized-compile problem. The set of kinds is closed and small.
- **Keep `make_*_toolchain` / `ToolchainConfig`**: naming public builders after an internal class
  while returning `CompiledBackend` perpetuates the vocabulary split.
- **Trim only dead combinators**: keeps shallow wrappers, reading cost and `Any`-typed composition;
  every ADR 0011 requirement survives full removal.
- **Per-call `translate()` options**: violates ADR 0011's composition-time reconfiguration rule.
