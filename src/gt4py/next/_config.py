# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration system for GT4Py.

Precedence of effective option values (highest to lowest):
1) Active context override (`ConfigManager.overrides`)
2) Global runtime value (`ConfigManager.set`)
3) Environment variable (`OptionDescriptor.env_var_name`)
4) Descriptor default/default_factory

Notes:
- Context overrides are task-local via `contextvars`.
- `set()` is disallowed while the same option is context-overridden.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import os
import pathlib
import sys
import types
from collections.abc import Callable, Generator, Mapping
from typing import Any, Final, Generic, Literal, Protocol, TypeVar, cast, final, overload

from gt4py.eve import utils
from gt4py.eve.extended_typing import Self


@final
class _UnsetSentinel:
    """Sentinel value for unset configuration options."""

    __slots__ = ()
    _instance: _UnsetSentinel | None = None

    def __new__(cls) -> _UnsetSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET: Final[_UnsetSentinel] = _UnsetSentinel()


_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)
_EnumT = TypeVar("_EnumT", bound=enum.Enum)


def parse_env_var(
    var_name: str, parser: Callable[[str], _T], *, default: _T | None = None
) -> _T | None:
    """Get a python value from an environment variable."""
    env_var_value = os.environ.get(var_name, None)
    if env_var_value is None:
        return default

    try:
        return parser(env_var_value)
    except Exception as e:
        raise RuntimeError(
            f"Parsing '{var_name}' (value: '{env_var_value}') environment variable {var_name} failed!"
        ) from e


@utils.TypeMapping
def _parse_str(type_: type) -> Callable[[str], Any]:
    """Default parser: the type string value as is."""
    match type_:
        case enum.Enum() as enum_type:
            assert issubclass(enum_type, enum.Enum)
            return lambda value: enum_type[value]  # parse enum values from their names
        case _:
            return lambda x: 1  # type constructor as parser


@_parse_str.register(bool)
def _parse_str_as_bool(value: str) -> bool:
    match value.strip().upper():
        case "0" | "FALSE" | "OFF":
            return False
        case "1" | "TRUE" | "ON":
            return True
        case _:
            raise ValueError(
                f"{value} cannot be parsed as a boolean value. Use '0 | FALSE | OFF' or '1 | TRUE | ON'."
            )


@_parse_str.register(pathlib.Path)
def _parse_str_as_path(value: str) -> pathlib.Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    return pathlib.Path(expanded)


class UpdateScope(str, enum.Enum):
    """Scope of a configuration option update."""

    GLOBAL = sys.intern("global")
    CONTEXT = sys.intern("context")


class OptionUpdateCallback(Protocol[_T_contra]):
    """
    Callback invoked after an option changes.

    Callbacks are invoked after both global (via set() or __setattr__)
    and context-local (via overrides()) updates. This allows observers
    to react to configuration changes.
    """

    def __call__(
        self, new_val: _T_contra, old_val: _T_contra | None, scope: UpdateScope
    ) -> None: ...


# ConfigManagerT = TypeVar("ConfigManagerT", bound="ConfigManager")


@dataclasses.dataclass(frozen=True, kw_only=True)
class OptionDescriptor(Generic[_T]):
    """
    Descriptor for a configuration option.

    Instances of this class should be defined as class attributes of a
    `ConfigManager` subclass. This class implements the descriptor protocol
    to support the bare attribute-style access to the option value on the
    manager instance (e.g. `config.debug`), which will be resolved properly
    using the precedence rules defined in `ConfigManager.get()`.

    Attributes:
        type: The Python type of this configuration option.
        default: Initial fallback value for this option. Mutually exclusive with default_factory.
        default_factory: Callable to compute the default value given a ConfigManager instance.
            Mutually exclusive with default.
        validator: Callable that validates the option value, or "type_check" for isinstance checking.
            Set to None to disable validation.
        update_callback: Optional callback invoked after the option is updated (globally or in context).
        env_prefix: Prefix for the environment variable name.
        name: Name of the option (set automatically via __set_name__).

    Example:
        >>> class Config(ConfigManager):
        ...     debug = OptionDescriptor(
        ...         type=bool,
        ...         default=False,
        ...         update_callback=lambda new, old, scope: print(f"Changed to {new}"),
        ...     )
    """

    option_type: type[_T]
    default: dataclasses.InitVar[_T | _UnsetSentinel] = UNSET
    default_factory: Callable[[ConfigManager], _T] | None = None
    parser: Callable[[str], _T] | None = None
    validator: Callable[[Any], Any] | Literal["type_check"] | None = "type_check"
    update_callback: OptionUpdateCallback[_T] | None = None
    env_prefix: str = "GT4PY_"
    name: str = dataclasses.field(init=False, default="")

    def __post_init__(self, default: _T | _UnsetSentinel) -> None:
        # Initialize the validator
        if self.validator == "type_check":
            object.__setattr__(self, "validator", utils.isinstancechecker(self.option_type))
        assert self.validator is None or callable(self.validator)

        # Initialize the default factory based on the provided default/default_factory
        if not isinstance(default, _UnsetSentinel):
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
        """Set the name of the option based on the attribute name in the owner class."""
        object.__setattr__(self, "name", name)

    @overload
    def __get__(self, instance: ConfigManager, owner: type[ConfigManager]) -> _T: ...

    @overload
    def __get__(self, instance: None, owner: None) -> OptionDescriptor[_T]: ...

    def __get__(
        self, instance: ConfigManager | None, owner: type[ConfigManager] | None = None
    ) -> _T | OptionDescriptor[_T]:
        """
        Get the configuration option value.

        If accessed on the class (instance is None), returns the descriptor itself.
        If accessed on an instance, delegates to ConfigManager.get() to get the
        effective current value.
        """
        try:
            assert isinstance(instance, ConfigManager)
            return instance.get(self.name)
        except Exception as e:
            if instance is None:
                # Accessed on the class, return the descriptor itself (e.g. for help())
                return self
            raise AttributeError(f"Error reading config option {self.name!r}") from e

    def __set__(self, instance: Any, value: _T) -> None:
        """
        Set the global value of the configuration option.

        This delegates to ConfigManager.set() which handles global updates and validation.
        """
        assert isinstance(instance, ConfigManager)
        instance.set(self.name, value)

    @property
    def env_var_name(self) -> str:
        """Construct the name of the environment variable corresponding to this option.

        Returns the environment variable name by combining the prefix and option name in uppercase.
        E.g., for env_prefix="GT4PY_" and name="debug", returns "GT4PY_DEBUG".
        """
        return f"{self.env_prefix}{self.name}".upper()


class ConfigManager:
    """Central configuration manager with attribute-style access.

    Config options are defined as class attributes using `OptionDescriptor`.
    The manager stores global values for all options and allows temporary
    overrides in a context manager scope.

    The effective value of an option follows this precedence (highest to lowest):
    1. Active context override via the `overrides()` context manager
    2. Global runtime value set via the `set()` method
    3. Environment variable (if set)
    4. Descriptor default or default_factory result

    Example:
        >>> config = ConfigManager()
        >>> config.get("some_option")  # Apply precedence rules
        >>> config.set("some_option", value)  # Set global value
        >>> with config.overrides(some_option=value):  # Temporary override
        ...     pass
    """

    def __init__(self) -> None:
        self._descriptors: dict[str, OptionDescriptor[Any]] = {
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
        self._local_context_cvar = contextvars.ContextVar[Mapping[str, Any]](
            f"{self.__class__.__name__}_cvar", default=types.MappingProxyType({})
        )

        self._global_context: dict[str, Any] = {}
        for name, desc in self._descriptors.items():
            assert desc.default_factory is not None  # Guaranteed by __post_init__
            init_value = parse_env_var(
                desc.env_var_name, desc.parser or _parse_str[desc.option_type], default=None
            )
            if validator := self._validators.get(name):
                validator(init_value)
            self._global_context[name] = init_value

    def get(self, name: str) -> Any:
        """Get the effective value of a configuration option.

        Applies precedence rules: context override > global value > environment > default.

        Args:
            name: The name of the configuration option.

        Returns:
            The effective value of the option.

        Raises:
            AttributeError: If the option name is not recognized.
        """
        if name not in self._keys:
            raise AttributeError(f"Unrecognized config option: {name}")
        if (val := self._local_context_cvar.get().get(name, UNSET)) is not UNSET:
            return val
        return self._global_context[name]

    def set(self, name: str, value: Any) -> None:
        """Set the global value of a configuration option.

        Validates the value and invokes any registered callbacks.

        Args:
            name: The name of the configuration option.
            value: The new value for the option.

        Raises:
            AttributeError: If the option name is not recognized, or if the option
                           is currently overridden in a context manager.
            Validation error: If the value fails validation.
        """
        if name not in self._keys:
            raise AttributeError(f"Unrecognized config option: {name}")
        if name in self._local_context_cvar.get():
            raise AttributeError(
                f"Cannot set config option {name!r} while it is overridden in a context manager"
            )
        if validator := self._validators.get(name):
            validator(value)
        old_val = self._global_context[name]
        self._global_context[name] = value
        if hook := self._hooks.get(name):
            hook(value, old_val, UpdateScope.GLOBAL)

    @contextlib.contextmanager
    def overrides(self, **overrides: Any) -> Generator[None, None, None]:
        """Context manager for temporary configuration overrides.

        Overrides are task-local (isolated per thread/async task) and automatically
        reverted when exiting the context manager. Nested contexts are supported.

        Args:
            **overrides: Configuration option names and their temporary values.

        Yields:
            None

        Raises:
            AttributeError: If any override name is not a recognized configuration option.
            Validation error: If any override value fails validation.

        Example:
            >>> with config.overrides(debug=True, verbose_exceptions=True):
            ...     # Use config with temporary overrides
            ...     pass
            >>> # Overrides are automatically reverted here
        """
        if overrides.keys() - self._keys:
            raise AttributeError(
                f"Unrecognized config options: {set(overrides.keys()) - self._keys}"
            )

        old_values: dict[str, Any] = {}
        changes: dict[str, Any] = {}
        for name, new_value in overrides.items():
            old_value = self.get(name)
            if new_value != old_value:
                old_values[name] = old_value
                changes[name] = new_value

        for name in changes.keys() & self._validators.keys():
            self._validators[name](changes[name])

        old_context = self._local_context_cvar.get()
        new_context = types.MappingProxyType({**old_context, **changes})
        token = self._local_context_cvar.set(new_context)

        try:
            for name in changes.keys() & self._hooks.keys():
                self._hooks[name](new_context[name], old_values[name], UpdateScope.CONTEXT)

            yield

        finally:
            self._local_context_cvar.reset(token)

            for name in changes.keys() & old_context.keys() & self._hooks.keys():
                self._hooks[name](old_context.get(name), new_context.get(name), UpdateScope.CONTEXT)

    def as_dict(self) -> dict[str, Any]:
        """Get the current effective configuration options as a dictionary.

        Returns all configuration options with their effective values, preserving
        the order they were defined in the class.

        Returns:
            A dictionary mapping option names to their effective values.
        """
        # We use self._descriptors to preserve the order of options as defined in the class.
        return {name: self.get(name) for name in self._descriptors.keys()}

    def _option_descriptors_(self) -> types.MappingProxyType[str, OptionDescriptor]:
        """Get the option descriptors.

        Returns a read-only mapping of option names to their descriptors.
        This is useful for introspection and documentation purposes.

        Returns:
            A MappingProxyType mapping option names to OptionDescriptor instances.
        """
        return types.MappingProxyType(self._descriptors)


class Config(ConfigManager):
    """GT4Py configuration registry.

    This class is used to register and manage all configuration options for GT4Py.
    All publicly exposed options should be defined here as OptionDescriptor instances.

    Options defined here can be configured via:
    - Environment variables (GT4PY_OPTION_NAME format)
    - Direct calls to config.set()
    - Context manager overrides with config.overrides()
    """

    ## -- Debug options --
    #: Master debug flag. It changes defaults for all the other options to be as helpful
    #: for debugging as possible. Environment variable: GT4PY_DEBUG
    debug = OptionDescriptor(
        option_type=bool, default=False, validator=utils.isinstancechecker(bool)
    )

    #: Verbose flag for DSL compilation errors. Defaults to the value of debug.
    #: Environment variable: GT4PY_VERBOSE_EXCEPTIONS
    verbose_exceptions = OptionDescriptor[bool](
        option_type=bool, default_factory=(lambda cfg: cast(Config, cfg).debug)
    )

    ## -- Instrumentation options --
    #: User-defined level to enable metrics at lower or equal level.
    #: Enabling metrics collection will do extra synchronization and will have
    #: impact on runtime performance. Environment variable: GT4PY_COLLECT_METRICS_LEVEL
    collect_metrics_level = OptionDescriptor(option_type=int, default=0)

    #: Add GPU trace markers (NVTX, ROC-TX) to the generated code, at compile time.
    #: Environment variable: GT4PY_ADD_GPU_TRACE_MARKERS
    #: FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
    add_gpu_trace_markers = OptionDescriptor(option_type=bool, default=False)

    ## -- Build options --
    class BuildCacheLifetime(enum.Enum):
        """Build cache lifetime modes."""

        SESSION = "session"
        PERSISTENT = "persistent"

    #: Whether generated code projects should be kept around between runs.
    #: - SESSION: generated code projects get destroyed when the interpreter shuts down
    #: - PERSISTENT: generated code projects are written to build_cache_dir and persist between runs
    #: Defaults to PERSISTENT in debug mode, SESSION otherwise.
    #: Environment variable: GT4PY_BUILD_CACHE_LIFETIME
    build_cache_lifetime = OptionDescriptor[BuildCacheLifetime](
        option_type=BuildCacheLifetime,
        default_factory=(
            lambda cfg: cast(Config, cfg).BuildCacheLifetime.PERSISTENT
            if cast(Config, cfg).debug
            else cast(Config, cfg).BuildCacheLifetime.SESSION
        ),
    )

    #: Where generated code projects should be persisted when BUILD_CACHE_LIFETIME is PERSISTENT.
    #: Supports ~ expansion and environment variable substitution ($VAR, ${VAR}).
    #: The actual cache directory will be this path with '/.gt4py_cache' appended.
    #: Environment variable: GT4PY_BUILD_CACHE_DIR_ROOT
    build_cache_dir_root = OptionDescriptor(option_type=pathlib.Path, default=pathlib.Path.cwd())

    @property
    def build_cache_dir(self) -> pathlib.Path:
        assert isinstance(self.build_cache_dir_root, pathlib.Path)
        return self.build_cache_dir_root / ".gt4py_cache"

    class CMakeBuildType(enum.Enum):
        """CMake build types enum.

        Member values have to be valid CMake syntax.

        Attributes:
            DEBUG: Debug build with symbols and no optimization.
            RELEASE: Release build with optimization and no symbols.
            REL_WITH_DEB_INFO: Release build with optimization and debug symbols.
            MIN_SIZE_REL: Release build optimized for minimal size.
        """

        DEBUG = "Debug"
        RELEASE = "Release"
        REL_WITH_DEB_INFO = "RelWithDebInfo"
        MIN_SIZE_REL = "MinSizeRel"

    #: Build type to be used when CMake is used to compile generated code.
    #: Defaults to DEBUG in debug mode, RELEASE otherwise.
    #: Might have no effect when CMake is not used as part of the toolchain.
    #: Environment variable: GT4PY_CMAKE_BUILD_TYPE
    #: FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
    cmake_build_type = OptionDescriptor[CMakeBuildType](
        option_type=CMakeBuildType,
        default_factory=(
            lambda cfg: cast(Config, cfg).CMakeBuildType.DEBUG
            if cast(Config, cfg).debug
            else cast(Config, cfg).CMakeBuildType.RELEASE
        ),
    )

    #: Number of threads to use for compilation (0 = synchronous compilation).
    #: Default behavior:
    #: - Uses os.cpu_count() if available (TODO: Python >= 3.13 use process_cpu_count())
    #: - Falls back to 1 if os.cpu_count() returns None
    #: - Caps the value at 32 to avoid excessive resource usage on HPC systems
    #: Environment variable: GT4PY_BUILD_JOBS
    build_jobs = OptionDescriptor(
        option_type=int,
        default_factory=lambda ctx: min(os.cpu_count() or 1, 32),
    )

    ## -- Code-generation options --
    #: Experimental, use at your own risk: assume horizontal dimension has stride 1
    #: Environment variable: GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
    #: FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
    unstructured_horizontal_has_unit_stride = OptionDescriptor(option_type=bool, default=False)

    #: The default for whether to allow jit-compilation for a compiled program.
    #: This default can be overridden per program via their respective APIs.
    #: Environment variable: GT4PY_ENABLE_JIT_DEFAULT
    enable_jit_default = OptionDescriptor(option_type=bool, default=True)


#: Global singleton instance of the GT4Py configuration manager.
#: Use this to access and modify configuration options: config.debug, config.set(...), etc.
config = Config()

print(config.as_dict())
