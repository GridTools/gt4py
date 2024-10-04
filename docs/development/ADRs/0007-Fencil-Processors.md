---
tags: []
---

# Fencil Processors

- **Status** deprecated
- **Authors**: Rico Häuselmann (@DropD)
- **Created**: 2022-06-22
- **Updated**: 2022-10-05

The backend registry was deleted and replaced by calling the function instead of passing the string key to `execute_fencil`. The previous de-facto concept of backend (any function that takes a program) has been split into two types of `program_processors`, `program_executors` and `program_formatters`. Care was taken to not complicate applying either of those to a program.

In contrast to "backend" we can clearly define a **program processor**: Any function that takes an IteratorIR representation of a program as a parameter along with all the inputs needed to call the program is a **program processor**. What kind of processor depends on the behaviour of that function.

This was done to

1. Give a single semantic meaning to calling a `@fendef` decorated function or FieldView program: executing the program.
1. Reduce the potential for circular imports.
1. Prepare for compiled backends integration, which needs to distinguish between these two (and possibly more) types of `program_processors`.

## Changelog

### 2022-10-05

- updated to reflect name change `fencil_processors` -> `program_processors`

## Context

### The fencil wrapper `__call__` used to have unclear semantics

Previously, any function with the signature `(itir.FencilDefinition, *args, **kwargs) -> Any` could be registered as a backend and called via the `__call__` method of the return type of the iterator or FieldView decorator. Semantically this meant that calling a program can not be expected to execute the program without first knowing what the "backend" method actually does.

This was useful for testing because it allowed parametrize tests with any kind of program processor (coupled with whether the program processor should execute the program and hence the result should be validated).

### The registry was not particularly useful and promoted circular imports

All of these program processors were registered as "backends" in a registry. At first glance this may have appeared helpful, since the module in which the processor function was defined could simply register it with a name, and from anywhere else it could be picked by that name. However, the registering action would only run on import of the "backend" module, so for this to work all of them had to be imported somewhere along the line before any of them could be used. This was done inside the `iterator` subpackage, which made it possible to run into circular imports when importing from `iterator` inside the "backend" module.

### Preparing for compiled backends integration

While working on integrating foreign language (non-python) backends (E.g. generating C++ code, compiling a python extension and loading it back) it turned out to be desirable to explicitly separate different kinds of endpoints for program processing. In the envisioned end form of that system creating a python extension, compiling it and loading it back is decoupled from the foreign language code generator. The fencil wrapper's `__call__` method therefore needs to distinguish between 1) something that can execute the program 2) something that generates code which can be turned into an executable python extension and 3) everything else like debugging utilities etc.

## Decision

### Clarifying nomenclature

Anything that doesn't execute a program, but instead generates text from it is now called a `program_formatter`.

### Clarifying the `__call__` semantics

The following was done to clarify the semantics of fencil wrapper `__call__` (and FieldView Program `__call__`), in the case of using a non-embedded backend: The `__call__` method now checks that the backend it was passed was decorated with `@program_processors.processor_interface.program_executor`. In addition another method was introduced on the fencil / Program wrappers (`.format_itir` / `.format_itir`) to cover all other currently existing program processors, as they all generate a text string from the program.

### The backend registry

The registry was removed for now. Any code using a backend by key was changed to import the backend function and use that instead.

### Preparing for later

In addition to the `program_executor` decorator, a `program_formatter` decorator was added. Both of these mark the decorated function so that different kinds of backends can be distinguished and they can be either directly executed or their output compiled, loaded back and then executed. Most likely that will involve an additional program processor kind with it's own decorator and entry point function.

### Preserving testability

In order to preserve the ease of testing any processor on all test cases, a dispatcher had to be added to the test suite, which allows that.

## Consequences

- Backends have to be explicitly imported before being used (used to be implicitly imported)
- Adding a true plugin architecture for backends has not become harder or easier
- Calling a program now always means executing it
- Adding a new kind of program processor and dispatching based on that kind is easy now

## Alternatives Considered

### Separating into categories of "executors" and "everything else"

The only use case that didn't either execute or generate a string was `pretty_print_and_check`. This was moved to the test suite because it would likely not be useful in practice. This step can be taken later if the variety general program processors ever increases.
