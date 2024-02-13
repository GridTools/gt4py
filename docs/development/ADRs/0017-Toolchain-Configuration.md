---
tags: [backend, otf, workflows, toolchain]
---

# Toolchain Configuration

- **Status**: valid
- **Authors**: Rico Häuselmann (@DropD)
- **Created**: 2024-02-13
- **Updated**: 2024-02-13

In order to provide a streamlined user experience, we attempt to standardize how users of GT4Py stencils can configure how those stencils are optimized without editing GT4Py code.

## Context

In this document the word toolchain is used to mean all the code components that work together to go from DSL code to an optimized, runnable python callable.
It includes JIT / OTF pipelines or workflows but also transformation passes, lowerings, parsers etc.

The most pressing issue is the developer experience. One debugging technique is to have the generated C++ code written to a permanent file location for inspection. This requires changing the code to reconfigure the build cache location. However this is forseeably only one of multiple values that stencil developers and their end users will wish to configure without touching GT4Py code.

At the time of creation of this document, at least one additional toolchain is actively being worked on and there are plans to make additional parts of the existing toolchains configurable, both of which will compound the issue.

**Concerns**

- Some ways of configuring toolchains involve configuring multiple components in synch
- Some ways of configuring toolchains involve switching out or nesting toolchain steps
- What configuration options are avaiable will depend on what toolchain will be used and how that is configured.
- Hierarchical configuration defaults and overrides can be confusing from a user perspective.
- Leaving the configuration interface completely up to toolchain developers could lead to a confusing ad fragmented user experience.

## Decision

### Opt-in pattern for building toolchains from user configuration and client code

Any toolchain that has user configurable options should provide a high level interface for building a toolchain that is consistent with the options set by the end user. If a default toolchain instance is provided in GT4Py code, it should use that interface. This ensures that the simplest way of obtaining an instance of a toolchain respects user configuration.

### Limit configuration options exposed to the end user

Any option that the end user can change in order to influence the toolchain behavior must be defined in `gt4py.next.config` with

- an internal name (a module level variable)
- an external name used to load from environment variables (possibly with a common prefix)
- a fallback default value in case no environment variable is defined

Any other toolchain option is considered an implementation detail.

### Limit the times when configuration can change

By making the `gt4py.next.config` module contain module level variables with user configuration, we ensure user configuration can only be changed between python interpreter runs (after `from gt4py import next` the configuration is fixed). Of course there are ways around it but they should be considered unsupported as they are difficult to make reliable (consider monkey patching as an example).

### Environment variables are the primary end user interface

Each user configurable option must be loaded from an environment variable if it is set. If there is an in-code default value it must be overridden by the environment variable.

## Consequences

### Testability is limited

Since the process of loading user configuration is in module scope, it is not repeatable for testing purposes. Module unloading and reloading is too much work and fragile.

Monkey patching can be used to test whether the values in the config module are reflected in newly built toolchains, but this very process sidesteps the actual reading of user configuration, which remains inaccessible to testing.

### Implementation is kept simple

All the logic for how to interpret the user configuration is manually encoded in the high level toolchain building interface. This does not currently require much code. No infrastructure is needed to automatically gather configuration options from toolchain definitions and present them to the user, as the exposed options are hand-picked.

Config variables can be assumed to be fixed for the entirety of a run of any program that uses GT4Py.

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

The first PoC used a function call to load user configuration just before using it. This would have increased testability. The only stated reason to switch to module level code is that `gt4py.cartesian` does it this way.

### Dynamical exposing of configuration options

It is in principle possible for every toolchain building interface to pick what it considers to look like configuration options from the environment variables. In practice this would make it very difficult to keep the experience consistent between toolchains.
