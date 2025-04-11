---
tags: [backend, otf, workflows, toolchain]
---

# Toolchain Configuration

- **Status**: valid
- **Authors**: Rico Häuselmann (@DropD), Till Ehrengruber (@tehrengruber)
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
- What configuration options are available will depend on what toolchain will be used and how that is configured.
- Hierarchical configuration defaults and overrides can be confusing from a user perspective.
- Leaving the configuration interface completely up to toolchain developers could lead to a confusing ad fragmented user experience.

## Decision

**All decisions below are in the spirit of keeping the scope of the initial implementation small and can be changed whenever changing them is suitably justified.**

### Opt-in pattern for building toolchains from user configuration and client code

Any toolchain that has user configurable options should provide a high level interface for building a toolchain that is consistent with the options set by the end user. If a default toolchain instance is provided in GT4Py code, it should use that interface. This ensures that the simplest way of obtaining an instance of a toolchain respects user configuration.

The pattern established for the 'GTFN' toolchain uses [`factory-boy`](https://factoryboy.readthedocs.io/en/stable/index.html), a package designed for constructing [ORM](https://en.wikipedia.org/wiki/Object%E2%80%93relational_mapping) models.

```python
class ToolchainFactory(factory.Factory):
    class Meta:
        model = ToolchainClass

    class Params:
        high_level_parameters = ... # check factoryboy docs for possibilities
        some_option = config.SOME_OPTION # default read from config module

    attribute_defaults = ... # may use parameter values etc
```

### Limit configuration options exposed to the end user

Any option that the end user can change in order to influence the toolchain behavior must be defined in `gt4py.next.config` with

- an internal name (a module level variable)
- an external name used to load from environment variables (possibly with a common prefix)
- a fallback default value in case no environment variable is defined

Any other toolchain option is considered an implementation detail from the point of view of the end user.

```python
# gt4py.next.config

#: information about the configuration option
INTERNAL_NAME_1 = os.environ.get(f"{_PREFIX}_EXTERNAL_NAME_1", <fallback default>)

#: information about the configuration option
INTERNAL_NAME_2 = os.environ.get(f"{_PREFIX}_EXTERNAL_NAME_2", <fallback default>)
```

Note that this module thus contains a handy list of all environment variables one can set to influence GT4Py behavior from the outside. It might be used to create the end user configuration documentation with sphinx, if variable docstrings are consistently used.

### Read end user configuration only once at import time

We design `gt4py.next.config` as a module with module level variables, which are initialized at import time from environment variables (if they exist).
We are aware that this decision has significant drawbacks. The main justification for it is to keep scope minimal by reusing the pattern from `gt4py.cartesian`.

```python
# gt4py.next.config
MASTER_SWITCH = os.environ.get(f"{_PREFIX}_MASTER_SWITCH", "false")
DEPENDENT_OPT = os.environ.get(f"{_PREFIX}_DEPENDENT_OPT", "one_default" if MASTER_SWITCH else "another_default")

if MASTER_SWITCH:
    ... # more complex config related side effects
```

### Environment variables are the primary end user interface

Each user configurable option must be loaded from an environment variable if it is set. If there is an in-code default value it must be overridden by the environment variable. This, particularly, was decided only to keep the implementation minimal.

Changing this without changing the import time initialization for adding a configuration file might look something like the following:

```python
# gt4py.next.config

_FILE_CONFIG = read_config_file()

OPTION1 = os.environ.get(f"{_PREFIX}_OPTION1", _FILE_CONFIG.option1 or "<fallback>")
```

## Consequences

### Changing configuration variables at runtime can lead to inconsistencies

Config variables are module-level and initialized at import time. Therefore any

- logic that switches one of them based on another or any other module-level
- initialization of dependent module-level defaults
- side effects

Will also have to happen at import time. At least in the first case it can **only** happen at import time.
This means changing the variables after import time will lead to inconsistencies if any of those patterns are present.

Implementations of the two latter patterns can be designed to mitigate this but at the cost of increased complexity elsewhere in the code. The first pattern can not.

```python
# gt4py.next.config
MASTER_SWITCH = os.environ.get(f"{_PREFIX}_MASTER_SWITCH", "false")
DEPENDENT_OPT = os.environ.get(f"{_PREFIX}_DEPENDENT_OPT", "one_default" if MASTER_SWITCH else "another_default")

if MASTER_SWITCH:
    ... # more complex config related side effects

# in client code
from gt4py.next import config

config.MASTER_SWITCH = "true"
# config.DEPENDENT_OPT has not been changed and the logic of how to change it is not accessible to be called at this point
```

#### Testability is limited as a result

- the patterns outlined above are not repeatable for testing purposes
- the potential resulting inconsistencies in configuration limit usefulness of changing the config variables for testing

The example above illustrates this, the test being the client code in this case.

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

```python
toolchain = ToolchainFactory(some_option="foo")  # this will override any `config.SOME_OPTION` default

@gtx.program(backend=toolchain)
def foo(...):
    ...
```

In this case the client program author has chosen to hardcode 'some_option', disregarding any 'config.SOME_OPTION' configuration variable read from the user environment. This must be documented clearly for the end user of the client code.

## Alternatives Considered

### Warn the user of workflow instances that disregard user configuration

The user would receive a warning, either

- when a toolchain instance is created, which disregards user config, or
- when such a toolchain is used

```python
toolchain = ToolchainFactory(some_option="foo")  # either at this point a warning would be issued:
# warning: 'client_code.toolchain' is overriding your configuration option 'GT4PY_SOME_OPTION'.

toolchain2 = ToolchainClass(...)  # it would be more effort to emit warnings when toolchains are constructed directly

@gtx.program(backend=toolchain)  # or at this point:
def foo(...):
    ...
# warning: Program 'client_code.foo' is using a toolchain that overrides your configuration option ...
```

The latter was never tried due to the obvious run time overhead in checking before every use. The former was experimented with and two types of implementation were considered: black-box analysis of the effects of configuration sources and configuration tracking.

Black-box analysis of the effects of different configuration sources would be far less intrusive and lightweight. In contrast to tracking, it can only follow configuration sources down to the toolchain construction entry point, not to lower level defaults. It was shown to be feasible, however we could not justify the maintenance burden. The following is a sketch of an algorithm for such analysis:

```python
env_defined_vars: dict[str, Any]  # This would contain the environment variables and all the dependent configuration variables
config_file_defined_vars: dict[str, Any]  # this would contain the configuration variables in the case of a clean environment
code_defined_vars: dict[str, Any]  # This would contain the parameters passed to the toolchain building entry point

var_sources = {
  "env": env_defined_vars,
  "config": config_file_defined_vars,
  "in-code": code_defined_vars
}

effects_per_var_source: dict[str, dict[str, list[str]]] = []

for var_source, vars in var_sources.items():
  res = factory(**vars)
  # compute what attributes are different in res from default, i.e. factory()
  effect = ...
  effects_per_var_source[var_source][var] = effect

for (source_a, effect_a), (source_b, effect_b) in itertools.product(effects_per_var_source.items(), effects_per_var_source.items())
  for name_outer, effect_outer in user_defined_vars_effects.items():
    for name_inner, effect_inner in code_defined_vars_effects.items():
      if intersection(effect_outer, effect_inner):
        print("{source_a}:{name_outer} conflicts with {source_b}:{name_inner}")
```

Note that:

- The current decision to initialize configuration variables at import time makes the distinction between `env_defined_vars` and `config_file_defined_vars` impractical.
- The algorithm would have to be inserted into every toolchain building entry point, which we want checked. This could be more or less high level and could be more or less involved.
- Logic for excluding conflicts between two levels of configuration which will be overridden anyway is not in the sketch but could be added.

As opposed to black-box analysis, tracking would mean to annotate each config variable with it's source. This would involve refactoring at every level down to each configurable toolchain component.
Implementing tracking was briefly considered but looked like it would be too heavy weight to justify the maintenance burden and too much work for the appetite of the implementation project. The current choice of high-level toolchain building pattern (`factory-boy`) does not particularly lend itself to implementing tracking.

### Dynamical loading of user configuration

The first PoC used a function call to load user configuration just before using it. This would have increased testability at the cost of a less minimal implementation.

```python

class Configuration:
    ...
    @property
    def option_1(self):
        ...

    @option_1.setter
    def option_1(self, value):
        self._option_1 = value
        self._dependent_option = ...
        ... # more dependent behavior

OPTION_1_DEFAULT

def get_configuration():
    conf = Configuration()
    conf.option_1 = read_from_env() or read_from_other_source() or default

current_configuration: Configuration = get_configuration()

# later on

config.current_configuration.option_1 = "foo"  # now all the dependent logic is handled correctly
```

A side effect of this would be that tests could work with independent configuration objects when necessary.

### Dynamical exposing of configuration options

It is in principle possible for every toolchain building interface to pick what it considers to look like configuration options from the environment variables. In practice this would make it very difficult to keep the experience consistent between toolchains.
