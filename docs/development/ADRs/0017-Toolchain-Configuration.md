---
tags: [backend, otf, workflows, toolchain]
---

# Toolchain Configuration

- **Status**: valid
- **Authors**: Rico Häuselmann (@DropD)
- **Created**: 2024-02-13
- **Updated**: 2024-02-13

In order to provide a streamlined user experience, we attempt to standardize how users of GT4Py stencils can configure how those stencils are optimized without editing GT4Py code. This describes the design of the first minimal implementation.

## Context

In this document the word toolchain is used to mean all the code components that work together to go from DSL code to an optimized, runnable python callable.
It includes JIT / OTF pipelines or workflows but also transformation passes, lowerings, parsers etc.

In this document the term "end user" refers to someone who runs an application which uses GT4Py internally. The end user may or may not be aware of GT4Py, only of the documentation that the application provides.

The most pressing issue is the developer experience. One debugging technique is to have the generated C++ code written to a permanent file location for inspection. This requires changing the code to reconfigure the build cache location. However this is forseeably only one of multiple values that stencil developers and their end users will wish to configure without touching GT4Py code.

At the time of creation of this document, at least one additional toolchain is actively being worked on and there are plans to make additional parts of the existing toolchains configurable, both of which will compound the issue.

**Concerns**

- Some ways of configuring toolchains involve configuring multiple components in synch
- Some ways of configuring toolchains involve switching out or nesting toolchain steps
- What configuration options are avaiable will depend on what toolchain will be used and how that is configured.
- Hierarchical configuration defaults and overrides can be confusing from a user perspective.
- Leaving the configuration interface completely up to toolchain developers could lead to a confusing ad fragmented user experience.

## Decision

**All decisions below are in the spirit of keeping the scope of the initial implementation small and can be changed whenever changing them is suitably justified.**

### Opt-in pattern for building toolchains from user configuration and client code

Any toolchain that has user configurable options should provide a high level interface for building a toolchain that is consistent with the options set by the end user. If a default toolchain instance is provided in GT4Py code, it should use that interface. This ensures that the simplest way of obtaining an instance of a toolchain respects user configuration.

The pattern established for the 'GTFN' toolchain uses `factory-boy`.

### Limit configuration options exposed to the end user

Any option that the end user can change in order to influence the toolchain behavior must be defined in `gt4py.next.config` with

- an internal name (a module level variable)
- an external name used to load from environment variables (possibly with a common prefix)
- a fallback default value in case no environment variable is defined

Any other toolchain option is considered an implementation detail from the point of view of the end user.

### Read end user configuration only once at import time

We design `gt4py.next.config` as a module with module level variables, which are initialized at import time from environment variables (if they exist).
We are aware that this decision has significant drawbacks. The main justification for it is to keep scope minimal by reusing the pattern from `gt4py.cartesian`.

### Environment variables are the primary end user interface

Each user configurable option must be loaded from an environment variable if it is set. If there is an in-code default value it must be overridden by the environment variable.

## Consequences

### Changing configuration variables at runtime can lead to inconsistencies

Config variables are module-level and initialized at import time. Therefore any

- logic that switches one of them based on another or any other module-level
- initialization of dependent module-level defaults
- side effects

Will also have to happen at import time. At least in the first case it can **only** happen at import time.
This means changing the variables after import time will lead to inconsistencies if any of those patterns are present.

Implementations of the two latter patterns can be designed to mitigate this but at the cost of increased complexity elsewhere in the code. The first pattern can not.

#### Testability is limited as a result

- the patterns outlined above are not repeatable for testing purposes
- the potential resulting inconsistencies in configuration limit usefulness of changing the config variables for testing

#### Implementation is kept minimal

Since we accept that changing the configuration at runtime may cause inconsistencies,

- we do not have to implement any way of delaying the point when we read the configuration to the last possible moment
- no new pattern needs to be established for how to expose end user config vars

### It is not possible to track where configuration values come from.

- the user-set environment variables must override the config module fallback value (can not be tested)
- the config module value must override the default values of the high-level toolchain building interface (testing up to toolchain implementer, requires monkey patching)
- the high-level toolchain building interface's defaults should override defaults in the toolchain modules (possibly testable, up to toolchain implementer)
- arguments passed to the high-level toolchain building interface should override all others, including user config (testing up to toolchain implementer)
- toolchain instances created without the high-level interface can do whatever they want (not testable in general)

The situation may arise that toolchain instances are used (possibly for good reason) in a program / library using GT4Py, which do not respect user configuration. It remains up to the implementer of such a program or library to communicate this to the user.

## Alternatives Considered

### Warn the user of workflow instances that disregard user configuration

The user would receive a warning, either

- when a toolchain instance is created, which disregards user config, or
- when such a toolchain is used

The latter was never tried due to the obvious run time overhead in checking before every use. The former turned out not to be very useful for two reasons:

- such a toolchain may have been created but not used to do work for the end user
- without proper tracking of where configuration comes from, false positives as well as false negatives could not be eliminated.

Implementing tracking was briefly considered but looked like it would be too heavy weight to justify the maintenance burden.

### Dynamical loading of user configuration

The first PoC used a function call to load user configuration just before using it. This would have increased testability at the cost of a less minimal implementation.

### Dynamical exposing of configuration options

It is in principle possible for every toolchain building interface to pick what it considers to look like configuration options from the environment variables. In practice this would make it very difficult to keep the experience consistent between toolchains.
