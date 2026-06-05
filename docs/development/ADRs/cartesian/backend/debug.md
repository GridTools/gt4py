# Debug backend

In the context of developing new DSL features, facing the question in which backend to prototype, we decided to implement a `debug` backend in python to allow for fast prototyping. We considered to prototype in other backends and accept the cost of having to maintain another backend.

## Context

The motivation behind the debug backend is twofold:

1. As DSL developers, we were missing an easy backend to prototype new DSL features. We find compiled performance backends not suitable for this task. And while the `numpy` backend outputs python code directly, the `numpy` code looks very different from the OIR representation.
2. As DSL (power) users / developers, we were missing a way to inspect how code is pushed through the system. For more complicated syntax like horizontal regions, it can be informative, to have a code output close to OIR.

## Decision

What is the change that we're proposing and/or doing?

We chose to implement the `debug` backend, which transparently maps OIR to plain python loops because this allows to easily prototype and inspect how DSL features are pushed through the stack.

## Consequences

The `debug` backend is a simple backend with a focus on readability of the generated python code. It is all python by design (e.g. no compile barrier), allows to visually inspect the generated code and simplifies things like setting a breakpoint inside a stencil. The simplicity of the backend is supposed to help with rapid feature development / prototyping.

On the flip side, the `debug` backend will be (painfully) slow to run even moderate sized examples. We chose readability over speed on purpose. The `numpy` backend is a python-based backend without this performance loss.

## Alternatives considered

### Prototype in existing backends

- Good, because we don't need an extra backend.
- Bad, because non of the backends are trivial to prototype. Compiled backends incur the cost of compilation and an explicit python \<-> C++ interface, which is detrimental for rapid prototyping of new DSL features. The `numpy` backend has to shape everything into a vector operation, obfuscating algorithmic design.
