---
tags: []
---

# [Argument Descriptors]

- **Status**: valid
- **Authors**: Till Ehrengruber (@tehrengruber)
- **Created**: 2025-09-12
- **Updated**: 2025-09-12

A generic mechanism to deduce (partial) information from runtime arguments and providing it at compile time is introduced.

## History

The first version of static parameters introduced a new class `CompiledProgramsPool` which was responsible for compiling and dispatching to a program compiled for a given set of arguments which were marked as static. How a static argument was fingerprinted in order to dispatch, how it was constructed to pass it down the toolchain was implemented directly inside the `CompiledProgramsPool` class. For static parameters this was a simple and working design, but adding additional information about arguments would have resulted in bloating the class with more and more code that was specific to the information we wanted to add: How do we construct it and from which subset of the value, how to fingerprint it without much overhead, how to validate the extracted data is correct, etc.

## Decision

We introduce a new class `ArgumentDescriptor` that all classes providing additional compile time information inherit from. This class contains all information needed to represent, extract and validate compile time information of an argument, hence uncoupling this from the `CompiledProgramsPool` implementation.

The `CompiledProgramsPool` class gets a new attributes `argument_descriptor_mapping` which maps from subclasses of `ArgumentDescriptor` to a list of parameter expression for which the respective descriptor is constructed for. We extend this from parameter names in the initial implementation to expressions such that we can, if desired, retrieve information for parts of the arguments (e.g. an element of a nested tuple). For static parameters this is less important, but usually we want a descriptor for a single leaf-type instead of a tuple / container, e.g. for a field instead of a tuple of fields.

In a first version we chose to allow multiple argument descriptors (internally) for a single parameter as this gives us maximum flexibility in the future. However, we will not expose this in the public facing interface, until we have actual cases where this is useful. Both approaches have their advantages and disadvantages. Having a single descriptor would mean we would need an additional mechanism to decide which information should be available to a descriptor, e.g. only the memory layout of a field or also its domain. Multiple descriptors allow distinguishing simply by not passing them in the `argument_descriptor_mapping`. On the other hand a single descriptor allows to describe a parameter with a single descriptor object when precompiling which reduces code bloat.

We chose expressions to describe which element of a parameter is used to construct argument descriptors as this is a convenient way to describe them for the user with a syntax that he is familiar with, e.g. one can just write `tuple_arg[0]` and because it was convenient in the implementation. Right now this is only used internally though and the user is only allowed to use parameter names.
