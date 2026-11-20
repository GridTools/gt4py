# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for importing modules lazily, on first attribute access."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Sequence
from typing import Any, Final, Literal, Optional, overload


_LAZY_MODULE_SLOT: Final = "_lazy_module_resolved_"
_LAZY_MODULE_LOCAL_ATTRS: Final = frozenset({"__name__", "__class__", _LAZY_MODULE_SLOT})


class _LazyModuleProxy(types.ModuleType):
    """Module stand-in importing the real module on first attribute access.

    The proxy is never registered in `sys.modules`. Resolution is a plain
    `importlib.import_module()` call, so import order, locking, side effects
    and failure rollback follow the standard import semantics. Only `__name__`
    and `__class__` are answered without importing; any other attribute access
    (including `__dict__`, `dir()`, attribute assignment and deletion) is
    forwarded to the real module.
    """

    def __init__(self, fullname: str) -> None:
        super().__init__(fullname)
        object.__setattr__(self, _LAZY_MODULE_SLOT, None)

    def _resolve(self) -> types.ModuleType:
        namespace = types.ModuleType.__getattribute__(self, "__dict__")
        module = namespace[_LAZY_MODULE_SLOT]
        if module is None:
            name = namespace["__name__"]
            module = importlib.import_module(name)
            # Do not cache a module which is still being initialized (e.g. when the proxy
            # is touched from the module's own import): if that import fails, plain
            # import semantics require a retry on the next access.
            # (`ModuleSpec._initializing` is an undocumented CPython attribute.)
            if sys.modules.get(name) is module and not getattr(
                getattr(module, "__spec__", None), "_initializing", False
            ):
                namespace[_LAZY_MODULE_SLOT] = module
        return module

    def __getattribute__(self, name: str) -> Any:
        if name in _LAZY_MODULE_LOCAL_ATTRS:
            return types.ModuleType.__getattribute__(self, name)
        try:
            module = _LazyModuleProxy._resolve(self)
        except ImportError as error:
            # Turn import errors into `AttributeError` for dunder probes (e.g.
            # `hasattr(obj, "__wrapped__")` in `inspect`), so introspection of
            # namespaces holding a proxy to a broken library does not crash.
            if len(name) > 4 and name.startswith("__") and name.endswith("__"):
                raise AttributeError(name) from error
            raise
        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_LazyModuleProxy._resolve(self), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(_LazyModuleProxy._resolve(self), name)

    def __dir__(self) -> list[str]:
        return dir(_LazyModuleProxy._resolve(self))

    def __repr__(self) -> str:
        namespace = types.ModuleType.__getattribute__(self, "__dict__")
        if (module := namespace[_LAZY_MODULE_SLOT]) is not None:
            return repr(module)
        return f"<lazy module '{namespace['__name__']}' (unloaded)>"


def _find_child_module_spec(
    name: str, search_path: Sequence[str]
) -> importlib.machinery.ModuleSpec | None:
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is not None and (spec := find_spec(name, search_path)) is not None:
            return spec
    return None


def _is_module_available(fullname: str) -> bool:
    """Check if `fullname` is importable without executing any module code.

    Parent packages already imported are inspected through their runtime
    `__path__`; the others through the search locations of their specs. If a
    child is not found below a parent which is not imported yet, the parent is
    imported for real and the lookup is retried, since its runtime `__path__`
    (or a `sys.modules` alias it creates) can differ from the static one.

    Raises:
        ImportError: if importing a parent package in the fallback path fails
            for any reason other than the module (or one of its parents) not
            being installed.
    """
    parts = fullname.split(".")
    search_path: Sequence[str] | None = None
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        is_last = i == len(parts)
        if name in sys.modules:
            module = sys.modules[name]
            if module is None:  # import explicitly blocked
                return False
            search_path = getattr(module, "__path__", None)
            if search_path is None and not is_last:  # not a package
                return False
            continue

        if i == 1:
            spec = importlib.util.find_spec(name)
        else:
            assert search_path is not None
            spec = _find_child_module_spec(name, search_path)
            parent_name = ".".join(parts[: i - 1])
            if spec is None and parent_name not in sys.modules:
                try:
                    importlib.import_module(parent_name)
                    # The parent can register the module directly in `sys.modules`
                    # (e.g. as an alias), possibly without a `__spec__`
                    if fullname in sys.modules:
                        return sys.modules[fullname] is not None
                    return importlib.util.find_spec(fullname) is not None
                except ModuleNotFoundError as error:
                    if error.name is not None and (
                        fullname == error.name or fullname.startswith(f"{error.name}.")
                    ):
                        return False
                    raise
        if spec is None:
            return False
        search_path = spec.submodule_search_locations
        if search_path is None and not is_last:  # not a package
            return False
    return True


@overload
def lazy_import_module(
    fullname: str, *, on_missing: Literal["raise"] = "raise"
) -> types.ModuleType: ...


@overload
def lazy_import_module(
    fullname: str, *, on_missing: Literal["none"]
) -> Optional[types.ModuleType]: ...


@overload
def lazy_import_module(fullname: str, *, on_missing: str) -> Optional[types.ModuleType]: ...


def lazy_import_module(fullname: str, *, on_missing: str = "raise") -> Optional[types.ModuleType]:
    """Return a proxy of a module which is only imported on first attribute access.

    The idea is the same as `lazy_loader.load()` from the scientific-python
    `lazy-loader` project, but it also works for subpackages (e.g.
    `"cupyx.profiler"`) without importing their parent packages: checking if
    the module is installed does not execute any module code, and the actual
    import happens on first attribute access, through a plain
    `importlib.import_module(fullname)` call.

    The returned proxy is not registered in `sys.modules` and it is not the
    real module object, so identity checks (`is`, `==`, dict keys) against
    the real module fail: use `resolve_lazy_module()` to get the real module
    in those cases. Other caveats:

    - Each subpackage needs its own call: `lazy_import_module("a")` only makes
      `a.b` reachable if `a` imports `a.b` itself (as with `import a`).
    - Only `__name__` and `__class__` are answered without importing. Probing
      other dunder attributes (e.g. `hasattr(proxy, "__wrapped__")`) imports
      the module, and an `ImportError` raised by that import is turned into
      an `AttributeError` (so `vars()` on a proxy of a broken library raises
      `TypeError`).
    - `importlib.reload()` does not accept proxies: use
      `importlib.reload(resolve_lazy_module(proxy))`.
    - Each attribute access goes through the proxy: in hot loops, bind the
      needed attributes or the result of `resolve_lazy_module()` to a local.

    Args:
        fullname: Absolute name of the module, e.g. `"cupy"` or `"jax.numpy"`.
        on_missing: What to do if the module, or one of its parent packages,
            is not installed: `"raise"` raises `ModuleNotFoundError`, `"none"`
            returns `None`. This only covers modules which are not installed:
            errors raised while importing an installed module (e.g. a GPU
            library without a working driver) are raised on first access.

    Returns:
        The real module if it is already imported, otherwise a proxy module,
        or `None` if the module is not installed and `on_missing` is `"none"`.

    Raises:
        ModuleNotFoundError: if the module is not installed and `on_missing`
            is `"raise"`.
        ImportError: if a parent package has to be imported to find the module
            (because it is not found statically) and that import fails.
        ValueError: if `on_missing` is not a valid option.

    Examples:
        >>> np_linalg = lazy_import_module("numpy.linalg")
        >>> int(np_linalg.det([[2.0, 0.0], [0.0, 2.0]]))
        4
        >>> lazy_import_module("not_installed_library", on_missing="none") is None
        True

        For type checkers, put the call in the `else` branch of an
        `if TYPE_CHECKING:` block containing a regular `import` of the module
        (type checkers then do not see that the result can be `None`).
    """
    if on_missing not in ("raise", "none"):
        raise ValueError(
            f"Invalid 'on_missing' value: expected 'raise' or 'none', got '{on_missing}'."
        )
    if (module := sys.modules.get(fullname)) is not None:
        return module
    if not _is_module_available(fullname):
        if on_missing == "none":
            return None
        raise ModuleNotFoundError(f"No module named '{fullname}'.", name=fullname)
    if (module := sys.modules.get(fullname)) is not None:  # imported by the availability check
        return module
    return _LazyModuleProxy(fullname)


def is_lazy_module(obj: Any) -> bool:
    """Check if `obj` is a proxy returned by `lazy_import_module()` (imported or not)."""
    return type(obj) is _LazyModuleProxy


def resolve_lazy_module(module: types.ModuleType) -> types.ModuleType:
    """Return the real module of a lazy module proxy, importing it if needed.

    Regular modules are returned unchanged.
    """
    if type(module) is _LazyModuleProxy:
        return _LazyModuleProxy._resolve(module)
    return module
