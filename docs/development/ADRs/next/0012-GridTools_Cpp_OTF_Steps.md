---
tags: [backend, cpp, gridtools, bindings, otf]
---

# GridTools CPP OTF Steps

- **Status**: valid
- **Authors**: Peter Kardos (@petiaccja), Rico Häuselmann (@DropD)
- **Created**: 2022-09-12
- **Updated**: 2022-09-14

## Context

This supersedes [0009 - Compiled Backend Integration](0009-Compiled_Backend_Integration.md) and concentrates on the on-the-fly compilation (OTFC) steps provided and / or used by the GridTools C++ backend.

## Decision

The main design decision was to implement the GridTools backend (GTFN) in terms of OTFC steps, which can be composed to workflows, with the exception of the executor, which instead uses a composed workflow to prepare execution.

## Consequences

As a consequence, the GTFN backend is now modular and some of it's logic can be shared with future backends. In addition it allows the user more control over which steps should be taken to fit their use case.

### Source Generator

The source generator part of the GTFN backend exposes two public interfaces.

#### Pure Source Code

`program_processors.formatters.gtfn.format_sourcecode` implements a `program_processors.processor_interface.ProgramFormatter` protocol and produces C++ source code from a GT4Py program and it's inputs. The resulting source code can not be called on it's own from GT4Py (or Python in general), but could be incorporated into a n external code base.

#### Translation Step

`program_processors.codegens.gtfn_modules.translate_program` implements the `otf.step_types.TranslationStep` protocol as well as `otf.workflow.Step` and produces a C++ source code module (source code packaged with additional information).

It is an instance of `program_processors.codegens.gtfn_modules.GTFNTranslationStep` with default language settings to be packaged into the source code module for later stages. These include the language used (C++), C++ dependencies, file name endings, code formatting options etc.

#### Binding Step

`otf.binding.pybind.bind_source` implements the `otf.step_types.BindingStep` protocol as well as `otf.workflow.Step` and generates an CompilableSource module from a source code module. An OTF module contains all the information required to compile the GT4Py program into a python callable. This includes bindings, in this case using `pybind11`. The bindings generator providing these is currently still tailored to the GTFN backend but should be easy to generalize.

#### Executor

`program_processors.runners.gtfn_cpu.run_gtfn` is the default `program_processors.processor_interface.ProgramExecutor` implementation of the GTFN backend. It sets sensible defaults for the configuration options on the individual steps. These include language settings and which builder should be used. Substep configuration can be achieved through passing one or more step instance to `dataclasses.replace(workflow.replace(run_gtfn.workflow, <name>=<instance>))`. Step instances can be obtained by replacing in a similar fashion.

## Alternatives Considered

### Monolithic Backend

In the earlier, Cartesian version of GT4Py backends started out as monoliths with just IR as input and executable program for output.

- Advantages:
  - in some cases simpler to write
  - not limited by inter-step interfaces, any information can be available at any step.
- Disadvantages:
  - hard to share code between backends
  - more design decisions required for implementing a new backend
  - leads to harder to maintain backends
  - steeper learning curve due to lack of consistency between backends

### Keep Bindings And / Or Build Steps Part Of GTFN

The bindings generator and builder steps could have been implemented without consideration for sharing code with future backends.

- Advantages:
  - faster to implement
- Disadvantages:
  - slows down the next backend implementation
