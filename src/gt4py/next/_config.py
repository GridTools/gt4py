# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import os
import pathlib
import sys
import types
from collections.abc import Callable, Generator
from typing import Any, Final, Generic, Literal, Protocol, TypeVar, cast, final


from gt4py.eve import utils
from gt4py.eve.extended_typing import Self


@final
class _UNSET_SENTINEL: ...


_UNSET: Final = _UNSET_SENTINEL()

_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)


@utils.type_dispatcher
def get_value_from_environment_var(
    as_type: type[_T], var_name: str, *, default: _T | None = None
) -> _T | None:
    """Convert the content of environment variable a typed value."""
    env_value = os.environ.get(var_name, None)
    if env_value is None:
        return default
    try:
        return as_type(env_value)
    except Exception as e:
        raise TypeError(
            f"Unsupported conversion of GT4Py environment variable {var_name}: {env_value}) to type '{as_type.__name__}'."
        ) from None


@get_value_from_environment_var.register(bool)
def _get_value_from_environment_var_as_bool(
    as_type: type[bool], var_name: str, *, default: bool | None = None
) -> bool | None:
    env_value = os.environ.get(var_name, None)
    if env_value is None:
        return default
    match env_value.upper():
        case "0" | "FALSE" | "OFF":
            return False
        case "1" | "TRUE" | "ON":
            return True
        case _:
            raise ValueError(
                f"Invalid GT4Py environment flag value for {var_name}: use '0 | FALSE | OFF' or '1 | TRUE | ON'."
            )


class UpdateScope(str, enum.Enum):
    GLOBAL = sys.intern("global")
    CONTEXT = sys.intern("context")


class OptionUpdateCallback(Protocol[_T_contra]):
    def __call__(
        self, new_val: _T_contra, old_val: _T_contra | None, scope: UpdateScope
    ) -> None: ...


ConfigRegistryT = TypeVar("ConfigRegistryT", bound="ConfigManager")


@dataclasses.dataclass(frozen=True, kw_only=True)
class OptionDescriptor(Generic[_T, ConfigRegistryT]):
    type: type[_T]
    default: dataclasses.InitVar[_T | _UNSET_SENTINEL] = _UNSET
    default_factory: Callable[[ConfigRegistryT], _T] | None = None
    validator: Callable[[Any], Any] | Literal["type_check"] | None = "type_check"
    update_callback: OptionUpdateCallback[_T] | None = None
    env_prefix: str = "GT4PY_"
    name: str = dataclasses.field(init=False)

    def __post_init__(self, default: _T | _UNSET_SENTINEL) -> None:
        if self.validator == "type_check":
            object.__setattr__(self, "validator", utils.isinstancechecker(self.type))
        assert self.validator is None or callable(self.validator)

        if default is not _UNSET:
            if self.default_factory is not None:
                raise ValueError(
                    "Cannot specify both default and default_factory for a config option descriptor."
                )
            if self.validator is not None:
                self.validator(default)
            object.__setattr__(self, "default_factory", lambda _: default)
        elif self.default_factory is None:
            raise ValueError(
                "Must specify either default or default_factory for a config option descriptor."
            )

    def __set_name__(self, owner: type, name: str) -> None:
        object.__setattr__(self, "name", name)

    def __get__(self, instance: Any, owner: type | None = None) -> _T | Self:
        try:
            assert isinstance(instance, ConfigManager)
            return instance.get(self.name)
        except Exception as e:
            if instance is None:
                # Accessed on the class, return the descriptor itself (e.g. for help())
                return self
            raise AttributeError(f"Error reading config option {self.name!r}") from e

    def __set__(self, instance: Any, value: _T) -> None:
        assert isinstance(instance, ConfigManager)
        instance.set(self.name, value)

    @property
    def env_var_name(self) -> str:
        return f"{self.env_prefix}{self.name}".upper()


class ConfigManager:
    """Central configuration registry with attribute-style access."""

    def __init__(self) -> None:
        self._descriptors: dict[str, OptionDescriptor[Any, Config]] = {
            name: attr
            for name, attr in type(self).__dict__.items()
            if isinstance(attr, OptionDescriptor)
        }
        self._keys = set(self._descriptors.keys())
        self._validators: dict[str, Callable[[Any], None]] = {
            name: desc.validator
            for name, desc in self._descriptors.items()
            if callable(desc.validator)
        }
        self._hooks: dict[str, OptionUpdateCallback[Any]] = {
            name: desc.update_callback
            for name, desc in self._descriptors.items()
            if desc.update_callback is not None
        }

        # An instance-level ContextVar creates isolated context-local state per manager
        # instance. Though discouraged in general (values bind to ContextVar identity
        # and Context objects hold strong references to ContextVars, so they won't be
        # GC'd even if the instance goes out of scope), in this case we really want
        # per-registry isolation and we assume only very few ConfigRegistry instances
        # will be ever created.
        self._local_context_cvar = contextvars.ContextVar[types.MappingProxyType](
            f"{self.__class__.__name__}_cvar", default=types.MappingProxyType({})
        )

        self._global_context: dict[str, Any] = {}
        for name, desc in self._descriptors.items():
            assert desc.default_factory is not None  # Guaranteed by __post_init__
            self._global_context[name] = get_value_from_environment_var(
                desc.type, desc.env_var_name, default=desc.default_factory(self)
            )

    def get(self, name: str) -> Any:
        if __debug__ and name not in self._keys:
            raise AttributeError(f"Unrecognized config option: {name}")
        if (val := self._local_context_cvar.get().get(name, _UNSET)) is _UNSET:
            return self._global_context[name]
        return val

    def set(self, name: str, val: Any) -> None:
        if __debug__ and name not in self._keys:
            raise AttributeError(f"Unrecognized config option: {name}")
        if name in self._local_context_cvar.get():
            raise AttributeError(
                f"Cannot set config option {name!r} while it is overridden in a context manager"
            )
        old_val = self._global_context[name]
        self._global_context[name] = val
        if hook := self._hooks.get(name):
            hook(val, old_val, UpdateScope.GLOBAL)

    @contextlib.contextmanager
    def overrides(self, **overrides: Any) -> Generator[None, None, None]:
        if __debug__ and overrides.keys() - self._keys:
            raise AttributeError(
                f"Unrecognized config options: {set(overrides.keys()) - self._keys}"
            )
        for name in overrides.keys() & self._validators.keys():
            self._validators[name](overrides[name])
        old_context = self._local_context_cvar.get()
        new_context = old_context | overrides

        token = self._local_context_cvar.set(new_context)

        try:
            for name in overrides.keys() & self._hooks.keys():
                self._hooks[name](
                    new_context[name],
                    old_context.get(name, self._global_context[name]),
                    UpdateScope.CONTEXT,
                )

            yield

        finally:
            self._local_context_cvar.reset(token)
            for name in overrides.keys() & old_context.keys() & self._hooks.keys():
                self._hooks[name](old_context.get(name), new_context.get(name), UpdateScope.CONTEXT)

    def as_dict(self) -> dict[str, Any]:
        """Get the current effective configuration options as a dictionary."""
        # We use self._descriptors to preserve the order of options as defined in the class.
        return {name: self.get(name) for name in self._descriptors.keys()}

    def _option_descriptors_(self) -> types.MappingProxyType[str, OptionDescriptor]:
        """Get the option descriptors."""
        return types.MappingProxyType(self._descriptors)


class Config(ConfigManager):
    """
    GT4Py configuration registry.

    This class is used to register configuration options for GT4Py.
    """

    ## -- Debug options --
    #: Master debug flag. It changes defaults for all the other options to be as helpful
    #: for debugging as possible.
    debug = OptionDescriptor(type=bool, default=False, validator=utils.isinstancechecker(bool))

    #: Verbose flag for DSL compilation errors.
    verbose_exceptions = OptionDescriptor[bool, "Config"](
        type=bool, default_factory=(lambda cfg: cast(bool, cfg.debug))
    )

    ## -- Instrumentation options --
    #: User-defined level to enable metrics at lower or equal level.
    #: Enabling metrics collection will do extra synchronization and will have
    #: impact on runtime performance.
    collect_metrics_level = OptionDescriptor(type=int, default=0)

    #: Add GPU trace markers (NVTX, ROC-TX) to the generated code, at compile time.
    # FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
    add_gpu_trace_markers = OptionDescriptor(type=bool, default=False)

    ## -- Build options --
    class BuildCacheLifetime(enum.Enum):
        SESSION = "session"
        PERSISTENT = "persistent"

    #: Whether generated code projects should be kept around between runs.
    #: - SESSION: generated code projects get destroyed when the interpreter shuts down
    #: - PERSISTENT: generated code projects are written to BUILD_CACHE_DIR and persist between runs
    build_cache_lifetime = OptionDescriptor[BuildCacheLifetime, "Config"](
        type=BuildCacheLifetime,
        default_factory=(
            lambda cfg: cfg.BuildCacheLifetime.PERSISTENT
            if cfg.debug
            else cfg.BuildCacheLifetime.SESSION
        ),
    )

    #: Where generated code projects should be persisted.
    #: Only active if BUILD_CACHE_LIFETIME is set to PERSISTENT
    build_cache_dir_root = OptionDescriptor(type=pathlib.Path, default=pathlib.Path.cwd())

    @property
    def build_cache_dir(self) -> pathlib.Path:
        assert isinstance(self.build_cache_dir_root, pathlib.Path)
        return self.build_cache_dir_root / ".gt4py_cache"

    class CMakeBuildType(enum.Enum):
        """
        CMake build types enum.

        Member values have to be valid CMake syntax.
        """

        DEBUG = "Debug"
        RELEASE = "Release"
        REL_WITH_DEB_INFO = "RelWithDebInfo"
        MIN_SIZE_REL = "MinSizeRel"

    #: Build type to be used when CMake is used to compile generated code.
    #: Might have no effect when CMake is not used as part of the toolchain.
    # FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
    cmake_build_type = OptionDescriptor[CMakeBuildType, "Config"](
        type=CMakeBuildType,
        default_factory=(
            lambda cfg: cfg.CMakeBuildType.DEBUG if cfg.debug else cfg.CMakeBuildType.RELEASE
        ),
    )

    #: Number of threads to use to use for compilation (0 = synchronous compilation).
    #: Default:
    #: - use os.cpu_count(), TODO(havogt): in Python >= 3.13 use `process_cpu_count()`
    #: - if os.cpu_count() is None we are conservative and use 1 job,
    #: - if the number is huge (e.g. HPC system) we limit to a smaller number
    build_jobs = OptionDescriptor(
        type=int,
        default_factory=lambda ctx: min(os.cpu_count() or 1, 32),
    )

    ## -- Code-generation options --
    #: Experimental, use at your own risk: assume horizontal dimension has stride 1
    # FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
    unstructured_horizontal_has_unit_stride = OptionDescriptor(type=bool, default=False)

    #: The default for whether to allow jit-compilation for a compiled program.
    #: This default can be overriden per program.
    enable_jit_default = OptionDescriptor(type=bool, default=True)


config = Config()

# if __name__ == "__main__":
#     print(aa)
#     self = sys.modules[__name__]
#     print(self.aa)
