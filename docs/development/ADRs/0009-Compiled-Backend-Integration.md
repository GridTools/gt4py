---
tags: [backend]
---

# Integration of compiled backends

- **Status**: superseded
- **Authors**: Peter Kardos (@petiaccja), Rico Häuselmann (@DropD)
- **Created**: 2022-07-13
- **Updated**: 2022-09-12

Summary of the key design choices made for the live execution of generated C++ (and other compiled) code.

**This document is superseded by**

- [0011 - On The Fly Compilation](0011-_On_The_Fly_Compilation.md), which talks about the general architecture and design of on the fly compilation, and
- [0012 - GridTools C++ OTF](0011-_GridTools_Cpp_OTF.md), which talks the specifics of on-the-fly compilation of the GridTools C++ backend generated code.

## Context

As per the current state, gt4py can execute iterator IR by generating Python code from it which is loaded back as a module. Additionally, gt4py can also emit equivalent C++ code from iterator IR. The integration of the compiled backends focuses on compiling the emitted C++ code and calling it from Python, effectively executing iterator IR live by translating it to machine code.

The process of executing ITIR this way consists of the following steps:

1. Generate C++ code equivalent to ITIR (_fencil code_)
2. Generate C++ code that wraps the fencil code and can be accessed from Python (_binding code_)
3. Compile the fencil and binding codes into machine code (a dynamically linked library)
4. Load the dynamic library into Python and extract the interfaces as Python objects
5. Call the interfaces from Python with the fencil's arguments

Step 1 is the core responsibility of a compiled backend. For interoperability between steps, the `fencil_processors.source_modules` subpackage provides utilities and data structures which should simplify implementation and guide design, without restricting the backend implementer (they are opt-in). Steps 2-4 should rely on `fencil_processors.builders` with little code required on the backend side. For backends where additional functionality is required, it is expected that the missing functionality be implemented inside those subpackages. Step 5 should typically simply relay the fencil arguments to the result of step 4 without additional code required.

The desired pipeline architecture for these steps is made easy to achieve by the provided library, on a per-backend level. The clear separation of the steps is not yet enforced, however, and neither is it completely implemented for the `gtfn` backend yet.

For reference, steps 2-4 are currently encoded in `fencil_processors.builders.callable.create_callable`, while examples for step 1 and 5 can be found at `fencil_processors.codegens.gtfn.gtfn_backend.generate` and `fencil_processors.runners.gtfn.run_gtfn` (which also encompasses all the other steps).

## Design

The chosen design is so that there is a clear interface for every step from ITIR to executable python extension, with the ability to proceed only as far as needed.

**Step 1:**

Translate IteratorIR into "backend language", for example C++ using GridTools.

The `SourceModuleGenerator` protocol in `fencil_processors.processor_interface` interface defines the interface for the first step, from IteratorIR to `fencil_processors.source_modules.source_modules.SourceModule`. This step can optionally make use of a `ProgramFormatter` (also defined in `processor_interface.py`) to deliver the source code along with information accessible to further steps.

The output of this step is a `SourceModule` instance, which is safely hashable.

This is the only step required if one is interested only in the backend language translation of the fencil (i.e. for integration into a non-python driven simulation).

**Step 2:**

Generate bindings to call the compiled backend language code from Python (Technically this could be for another language but then the pipeline would not lead to an executable fencil).

The interface for step two is defined in the `fencil_processors.pipeline.BindingStep` protocol. The first example is implemented in `fencil_processors.builders.cpp.bindings.create_bindings`.

The output of this step is a `BindingsModule` instance and also safely hashable.

This would could be useful as an endpoint to use the Python bindings as an example for handcrafted bindings to other languages or for distribution of the generated bindings in a self-contained library with it's own build system.

**Step 3a:**

Use an implementation of the `fencil_processors.pipeline.BuildableProjectGenerator` interface to obtain an object that fulfills the `fencil_processors.pipeline.BuildableProject` protocol.

The resulting object may be used to write a self-contained folder that contains all the information to build the bindings.

**Step 3b:**

Use the `fencil_processors.pipeline.BuildableProject.get_fencil_impl` method to obtain a callable that can execute the fencil given the inputs. Depending on the underlying build system of the concrete implementation, this may call additional steps when needed (such as writing to file, configuring, building etc).

## Python bindings for C++ code

### Alternatives

The most popular libraries to make C++ code accessible from Python are:

- **pybind11**
- Boost.Python
- Python/C & ctypes

### Decision

Python/C is a good choice if we want to have a C interface to the compiled code, which can then be called not only from Python, but from pretty much any other language. On the other hand, having to manually handle C++ types and exceptions makes this approach much more complicated.

Boost.Python and pybind11 are very similar. Since boost is not a requirement for the bindings themselves (though it is for all current C++ fencil code), it makes sense to use the much more lightweight and perhaps more modern alternative of pybind11. Unlike Python/C, pybind11 handles STL containers and name mangling out of the box, and it also has a much more concise interface.

## Interface between binding and fencil code

### Alternatives

- Slim interface: requires implementing a tailored binding code generator for every fencil code generator
- **Universal interface**: requires all fencil code generators of the same technology (i.e. all C++ generators) to have the same interface. This allows one to write a single binding generator for C++ that can be paired with any fencil code generator

### Decision

We chose the universal interface for the following advantages:

- Promotes code reuse: slim bindings for different C++ backends would share a large fraction of their code
- Decoupling fencil generators from binding generators allows testing the two components separately
- Forces the separation of responsibilities between binding and fencil code generation

However there are some downsides and implementation issues:

- Although the proper architecture is in place, the current implementation still leaves the fencil and binding code somewhat intertwined
- Currently only two C++ backends are foreseen: GridTools CPU and GridTools CUDA. These backends have close to or completely identical C++ code. This situation might make code reuse less significant.

## Connection to FieldView frontend

We decided to keep compiled backend completely independent of the fields view frontend. The input the compiled backends is thus iterator IR and additional information necessary for code generation.

Reasons:

- There may be multiple frontends, but all of them would emit iterator IR and therefore would be compatible with compiled backends out of the box
- The FieldView frontend must import compiled backends. If compiled backends were aware of FieldView features, that would lead to a circular dependency between the frontend and backend.
- The work to support starting from iterator IR directly would have to be done in any case.

## Limitations

The main goal of this project is to implement the complete pipeline from FieldView to machine code and demonstrate that it's working. To keep the scope of the project reasonable, feature completeness is not targeted.

### Build system project

The `fencil_processors.builders.cpp.build.CMakeProject` class design should not be considered final because of the following pitfalls:

- the blocking `configure` and `build` functions may need to be converted to asynchronous operations to support parallel compilation of multiple fencils
- support needs to be added to switch between debug and release builds, as well as to conditionally enable debug information for release builds
- support needs to be added to enable compiler optimizations and tuning for target hardware

Considering these, future changes will likely require a more extended refactoring pass.

### Splitting existing fencil and binding code generators

The "roundtrip" backend could benefit from being split into a Python fencil generator and a small binding generator, which would merely write out the file and load it back live.

This is out of the scope of this project, but should be done in the future. Additional backends, wherever possible, should follow the separated pattern for clarity.

### Fencil argument types

In this implementation, only fencil having scalar and field arguments can be executed with compiled backends. Constant fields and index fields are to be added later.

### GridTools CUDA support

To keep things simple, GridTools CUDA is not supported in this first iteration.

To add support, we have to decide on the interface that is exposed to the user:

1. There are two completely separate backends for CPU and CUDA
2. There is a single backend and the device is selected by a flag

The existing code can be extended for either options in the future.

### Unstructured grids

This implementation only supports Cartesian grids, again, to keep things simple. Unstructured grids are to be added as soon as possible, however, that requires more design when it comes to passing the connectivities from Python to C++.

### Library dependencies

Compiled backends may generate code which depends on libraries and tools written in the language in which the backend generates code. There are three different categories currently, each occurring in the `gtfn` backend. They are libraries and tools, which

1. can be installed with `pip` (from `PyPI` or another source) automatically.
2. can not be installed with `pip` and not commonly found on HPC machines.
3. libraries and tools which are left to the user to install and make discoverable: `boost`, C++ compilers

Category 1 are made dependencies of `GT4Py`. Examples include `pybind11`, `cmake`, `ninja`.

The core library the backend is based on typically falls into Category 2. The only one currently is `GridTools`, which is downloaded and installed as part of the build process for each fencil. This is not part of the design but an implementation detail which should be changed (for example by moving `GridTools` into Category 1)

Category 3 contains compilers for the backend's language and libraries like `boost`, `mpi`, `CUDA` etc. They are currently left up to the user to deal with. In the long run user experience could be improved by providing a package in an appropriate package manager, which resolves these dependencies automatically.
