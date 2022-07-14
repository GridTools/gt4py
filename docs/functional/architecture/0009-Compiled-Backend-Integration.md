# Integration of compiled backends

- **Status**: valid 
- **Authors**: Peter Kardos (@petiaccja)
- **Created**: 2022-07-13
- **Updated**: 2022-07-13

Summary of the key design choices made for the live execution of generated C++ (and other compiled) code.

## Context

As per the current state, gt4py can execute iterator IR by generating Python code from it which is loaded back as a module. Additionally, gt4py can also emit equivalent C++ code from iterator IR. The integration of the compiled backends focuses on compiling the emitted C++ code and calling it from Python, effectively executing iterator IR live by translating it to machine code.

The process of executing ITIR this way consists of the following steps:
1. Generate C++ code equivalent (*fencil code*) to ITIR
2. Generate C++ code that wraps the fencil code and can be accessed from Python (*binding code*)
3. Compile the fencil and binding codes into machine code (a dynamically linked library)
4. Load the dynamic library into Python and extract the interfaces as Python objects 
5. Call the interfaces from Python with the fencil's arguments


## Python bindings for C++ code

### Alternatives

The most popular libraries to make C++ code accessible from Python are:

- **pybind11**
- Boost.Python
- Python/C & ctypes

### Decision

Python/C is a good choice if we want to have a C interface to the compiled code, which can then be called not only form Python, but from pretty much any other language. On the other hand, having to manually handle C++ types and exceptions makes this approach much more complicated.

Boost.Python and pybind11 are very similar. Since boost is not a requirement for the bindings themselves (though it is for all current C++ fencil code), it makes sense to use the much more lightweight and perhaps more modern alternative of pybind11. Unline Python/C, pybind11 handles STL containers and name mangling out of the box, and it also has a much more concise interface.

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


## Connection to field view frontend

We decided to keep compiled backend completely independent of the fields view frontend. The input the compiled backends is thus iterator IR and additional information necessary for code generation.

Reasons:
- There may be multiple frontends, but all of them would emit iterator IR and therefore would be compatible with compiled backends out of the box
- The field view frontend must import compiled backends. If compiled backends were aware of field view features, that would lead to a circular dependency between the frontend and backend.


## Limitations

The main goal of this project is to implement the complete pipeline from field view to machine code and demonstrate that it's working. To keep the scope of the project reasonable, feature completeness is left for future work.

### Splitting existing fencil and binding code generators

The roundtrip backend could benefit from being split into a Python fencil generator and a small binding generator, which would merely write out the file and load it back live.

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

This implementation only supports cartesian grids, again, to keep things simple. Unstructured grids are to be added as soon as possible, however, that requires more design when it comes to passing the connectivities from Python to C++.

### Library dependencies

In this iteration, library dependencies (i.e. boost, GridTools, pybind11) are handled in an inefficient and ad-hoc manner. To improve performance and user-friendliness, this should be resolved in future work.