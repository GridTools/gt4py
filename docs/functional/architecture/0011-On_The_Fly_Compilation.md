---
tags: [backend,bindings,build,compile,otf]
---

# On The Fly Compilation

- **Status**: valid
- **Authors**: Rico Häuselmann (@DropD)
- **Created**: 2022-09-12
- **Updated**: 2022-10-05

This supersedes [0009 - Compiled Backend Integration](0009-Compiled_Backend_Integration.md) and concentrates on the API design for on-the-fly compilation of stencils and all the steps in between IR and compiled Python extension.

## Context

The on-the-fly compilation (OTFC) in gt4py encompasses everything necessary to go from an IR representation of a stencil program (aka fencil) to an executable Python function. Depending on the chosen route, this may include:

- generating source code (in any language), including source code required to call other generated code from Python (if applicable)
- writing generated source code to the file system and subsequentely
  - compiling and / or
  - importing it back into Python
- decorating resulting Python constructs (wrapping)
- short-circuiting any or all of the above if the resulting Python constructs are already available

```
DSL -> IR -> Generate source -> [Bindings -> Write to FS -> Compile] -> Import -> Decorate -> Execute program
          |-------------------------------------------------------------------------|
                    GT4Py On-The-Fly Compilation
```

The main use case is to execute GT4Py stencil programs in a performance portable way from Python. Other usecases may include generating performance portable code to be integrated in external applications, generating code for debugging purposes, pre-compiling libraries of stencil programs for use from separate Python applications etc.

Much of the goals and the resulting architecture comes from lessons learned in GT4Py Cartesian, where originally all the steps of on-the-fly compilation together with code generation were approached as one monolithic implementation per "backend" (method of generating external source files). This had to be refactored to account for some use cases and lead to tight coupling with small changes rippling though more code than necessary.

## Naming

To avoid confusion the commonly occuring stages of OTF workflows have names (some of which map to protocols).

**`OTFClosure`:**
IR representation of a GT4Py program wrapped together with the arguments that will be passed to it.

**`SourceModule`:**
Backend source code wrapped together with information on how it wants to be called.

**`OTFSourceModule`:**
`SourceModule` wrapped together with optional language bindings (source code that enables the backend source code to be called from another language).

**Executable Program:**
A python object that executes the GT4Py program when called with the same arguments as stored in the `OTFClosure`.

Similarly the steps to go from one of the above stages to another have names.

**`ProgramSourceGenerator`:**
`OTFClosure -> SourceModule`.

**`PackagingStep`:**
`SourceModule -> OTFSourceModule`.

**Bindings Generator:**
A special case of `PackagingStep` where the resulting `OTFSourceModule` contains language bindings.

**BuildSystemProject:**
`OTFSourceModule -> Executable Program`.

## Architecture

The goals of the on-the-fly compilation (OTFC) architecture and design are:
- guide and inform the design and implementation of future components in this part of the code (code generators, builders, bindings generators, caching strategies etc)
- ensure consistency and readability of components
- keep components easy to reason about
- keep components (including future ones) intercompatible as much as reasonably possible
- allow per-usecase flexible composition of components

The chosen architecture is composable workflows. Such workflows can be composed of workflow steps, which implement the workflow step protocol (@todo: add reference), provided each step's input argument type(s) matches the preceeding step's return type(s). Any step customization should happen during workflow composition and *not* by passing options and flags along the workflow.

The completed workflow can then be called with the input argument(s) of the first workflow step and it's return type will be the same as the last step's.

## Design Decisions

### Linear Workflows
- decided: 2022-09-13
- revised: --

For the first implementation, workflows are chosen to be linear. That is to say each step is followed by exactly zero or one step. No branching, step-skipping or short-circuiting is implemented.

#### Reason
Not required at this stage and simplifies implementation

#### Consequences
Caching is therefore currently not implemented as a separate step and is therefore the concern of other steps (currently only the build step).

#### Revise If
There is a good usecase that justifies the extra effort. Such as implementing caching steps as conditionals based on which subsequent steps are switched or skipped, or which might complete the workflow early.

### Statically Typed Workflows
- decided: 2022-09-13
- revised: --

Workflow steps are generic on input and output type. Workflows are generic on input, intermediary and output type. This allows the use of static typecheckers to ensure that the composed workflow has the intended input and output types, as well as that it is passed the correct argument types and that it's return value is used correctly.

In order to achieve this, workflows are implemented as nested pairs (or more generally, finite sized tuples) of steps, with the output type of the first step matching the input type of the second step in the pair. More complex workflow components than pairs might be introduced if the "Linear Workflows" decision is revised.

To allow building workflows in a linear-looking way in code, the first step can be wrapped in a <@todo: add name of workflow step class for this> instance, on which `.chain` can be called with the next step. The result will be a workflow instance with `.chain` available also.

```
workflow = <@todo put class name here too>(step1).chain(step2).chain(step3).chain(step4)
# results in
Workflow(
  first=Workflow(
    first=Workflow(
      first=<@todo class name>(step1),
      second=step2
    ),
    second=step3
  ),
  second=step4
)
```

#### Reason
It is in line with the rest of GT4Py to make static type checking for it's code objects as useful as possible. Any linear (and with extensions, non-linear) workflow can be represented in this way.

#### Revise If
There has to be a very strong reason to drop static type checking for composition and / or the resulting workflow. The author of this document can not imagine one at present.

### Shared Bindings Step
The workflow step from the `SourceModule` stage to the `OTFSourceModule` stage, adding python bindings via `pybind11` was designed to be shared between C++ backends. In practice it is not quite there yet, but close enough for the current stage of development.

The same decision should be followed for other language bindings, wherever possible.

#### Consequences
The `program_processors.source_modules.source_modules.SourceModule` class has been designed to carry enough information for the bindings generator to convert from the common python call interface to whatever the backend C++ code requires.

#### Reason
The advantage is that compiled programs can be called with the same arguments independently of which (C++) backends were used, without forcing backend implementers to reinvent the wheel when it comes to converting to data structures their bakend's C++ side knows how to work with.

#### Revise If
Should it turn out that future backends are not sufficiently similar to warrant sharing this step, it may make sense to specialize these for each C++ backend instead. Most likely some code will still be shareable between these though.

### Shared Build Steps
The build step goes from `program_processors.source_modules.source_modules.OTFSourceModule` to an executable python object. Two versions have been implemented in `program_processors.builders.cpp.cmake` and `program_processors.builders.cpp.compiledb`. 

`cmake.CMake` creates a new full CMake project for each program. `compiledb.Compiledb` creates a new CMake project only if it would lead to a different CMake configuration, such as diverging dependencies or build types etc.

These can be extended to cover more than C++ but this should be done when such usecases arise, similar to the `pybind11` bindings step.

## Alternatives Considered

### Pipeline Architecture
A pipeline architecture was considered, but this would require the input and outpu argument of every step be the same. This would lead to a data type with many states, able to represent anything from source code to executable stencil programs. This was not considered wise.

### Ad-hoc Workflows Through Function Composition
At first, workflows were to be simply composed by the result of each step being passed as an argument to the following one. The resulting code was deemed hard to read and, potentially, to maintain. Example:

```python=
result = step3(
    option1=...,
    option2=...
    inp=step2_factory(
      option3=...,
      option4=...,
    )(
      inp=step1(...)
    )
)
```

Note that the steps occur in reverse order when the code is read top to bottom. In addition the customization of steps can easily get mixed in with the *flow* of the workflow. Finally it is common to refactor code like that for readability to something like:

```python=
intermediary_1 = step1(...)
intermediary_2 = step2(option3=..., option4=...)(inp=intermediary_1)
result = step3(option1=..., option2=..., inp=intermediary_2)
```

Which relies solely on the discipline of all contributors to not introduce logic between steps, whithout making that logic available as a workflow step. Hence the maintainability issue.

### Managing Workflow Steps In A Sequence

It looks enticing to be able to create a linear workflow by passing a (non-finite sized) sequence of steps:

```python=
workflow = Workflow(step1, step2, step3, ...)
# or
workflow = workflow_from_list([step1, step2, step3, ...])
```

However this disables static type checking, as there are no variadic generics in python (yet?).
