# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for importing modules lazily, on first attribute access.

This module is a temporary stand-in for the explicit lazy imports of
Python 3.15 (PEP 810, e.g. `lazy import cupyx.profiler as profiler`), which
are not available in the Python versions supported by GT4Py (3.12-3.14). Its
semantics deliberately follow PEP 810, so call sites can be migrated to the
`lazy` keyword once Python 3.15 is the minimum supported version:

- The module is not executed, nor added to `sys.modules`, until its first
  use (unless finding it requires importing a parent package, see below).
  It is then imported with `importlib.import_module()`, so import order,
  side effects and errors are those of a regular import statement.
- If the import fails, nothing is cached and the next use retries it.

Differences with PEP 810 that call sites must take into account (the first
two go away after migrating):

- The proxy object is not replaced by the real module after the first use
  (PEP 810 rebinds the name), so identity-based checks (`is`, `==`, `in`,
  dict keys, or utilities checking that an object is importable under its
  qualified name) need `resolve_lazy_module()`.
- Every attribute access goes through the proxy, also after the first use.
- Availability is checked eagerly: missing modules can be turned into `None`
  (`on_missing="none"`). PEP 810 forbids lazy imports inside `try` blocks, so
  optional dependencies will still need an explicit availability check before
  a `lazy import`. Note that `importlib.util.find_spec()` imports the parent
  packages of dotted names, while the check implemented here only does it
  when the module cannot be found statically (e.g. it is only reachable
  through the runtime `__path__` of a parent, or registered by its parent in
  `sys.modules`); in that case the parents run at call time, not at first use.

The PEP 810 `__lazy_modules__` module global (a collection of module names
whose imports become lazy on Python 3.15+, ignored by older versions) can be
used as an intermediate migration step for modules which are always
installed.

Design notes: unlike `lazy_loader.load()` (scientific-python `lazy-loader`)
or `importlib.util.LazyLoader`, the proxy is never registered in
`sys.modules`. A lazy module object in `sys.modules` is loaded by any scan of
it (e.g. `inspect.getmodule()`, which GT4Py itself triggers at import time),
makes an installed-but-broken library raise its `ImportError` inside
unrelated code probing `sys.modules`, and, for subpackages, can run a child
module before its parent package.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Sequence
from typing import Any, Final, Literal, Optional, overload


# Sentinel for `sys.modules` lookups (single lookups, since other threads can modify it)
_NOT_FOUND: Final = object()

# Name of the proxy attribute caching the resolved module. It is always answered by
# the proxy itself, so it shadows an attribute with the same name in the real module.
_LAZY_MODULE_SLOT: Final = "_lazy_module_resolved_"
_LAZY_MODULE_LOCAL_ATTRS: Final = frozenset({"__name__", "__class__", _LAZY_MODULE_SLOT})


class _LazyModuleProxy(types.ModuleType):
    """Module stand-in importing the real module on first attribute access.

    The proxy is never registered in `sys.modules`. Resolution is a plain
    `importlib.import_module()` call, so import order, locking, side effects
    and failure rollback follow the standard import semantics. Only `__name__`
    and `__class__` (and the private attribute caching the resolved module)
    are answered without importing; any other attribute access
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
            # Turn `ImportError`s into `AttributeError`s for dunder probes (e.g.
            # `hasattr(obj, "__wrapped__")` in `inspect`), so introspecting a
            # namespace holding a proxy to a library which fails to import does not
            # crash. Other exceptions are propagated unchanged, to avoid masking bugs.
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
        if find_spec is None:
            continue
        try:
            spec = find_spec(name, search_path)
        except (ImportError, KeyError):
            # Some finders need the parent package to be imported, e.g. `PathFinder`
            # for a namespace package portion below a parent missing in `sys.modules`
            # (`KeyError` up to Python 3.14, `ModuleNotFoundError` afterwards): report
            # it as not found so the caller falls back to importing the parent.
            return None
        if spec is not None:
            return spec
    return None


def _import_parent_and_check(parent_name: str, fullname: str) -> bool:
    """Import the parent of `fullname` for real and check if `fullname` is importable.

    The runtime `__path__` of the parent, or `sys.modules` entries it creates
    (e.g. aliases, also below non-package modules), can differ from what is
    statically visible. Note that `importlib.util.find_spec(fullname)` also
    imports any intermediate packages between the parent and `fullname`.
    """
    try:
        importlib.import_module(parent_name)
        # Registered by the parent (e.g. as an alias, maybe without `__spec__`)
        if (module := sys.modules.get(fullname, _NOT_FOUND)) is not _NOT_FOUND:
            return module is not None
        return importlib.util.find_spec(fullname) is not None
    except ModuleNotFoundError as error:
        if error.name is not None and (
            fullname == error.name or fullname.startswith(f"{error.name}.")
        ):
            return False
        raise


def _is_module_available(fullname: str) -> bool:
    """Check if `fullname` is importable, executing module code only if needed.

    Parent packages already imported are inspected through their runtime
    `__path__`; the others through the search locations of their specs, which
    does not execute any module code. Only if the module cannot be found this
    way below a parent which is not imported yet (or the parent is not a
    package), the parent is imported for real and the lookup is retried, since
    its runtime `__path__` or the `sys.modules` entries it creates can differ
    from the static view (see `_import_parent_and_check()`).

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
        if (module := sys.modules.get(name, _NOT_FOUND)) is not _NOT_FOUND:
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
            if spec is None:
                return _import_parent_and_check(".".join(parts[: i - 1]), fullname)
        if spec is None:
            return False
        search_path = spec.submodule_search_locations
        if search_path is None and not is_last:
            # Not a package, but its code can still register submodules in `sys.modules`
            return _import_parent_and_check(name, fullname)
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

    Temporary equivalent of `lazy import <fullname>` (PEP 810, Python 3.15+):
    see the module docstring for the differences and the migration path. It
    also works for subpackages (e.g. `"cupyx.profiler"`): checking if the
    module is installed does not execute any module code when the module can
    be found through the search locations of its parent packages' specs (the
    usual case), and the actual import happens on first attribute access,
    through a plain `importlib.import_module(fullname)` call. Otherwise (e.g.
    for modules only reachable through a parent's runtime `__path__`, or
    registered by a parent in `sys.modules`), the parent packages are
    imported by this call to find out.

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
        The real module if it is already imported (also when importing a
        parent package to find it imported it), otherwise a proxy module, or
        `None` if the module is not installed and `on_missing` is `"none"`.

    Raises:
        ModuleNotFoundError: if the module is not installed and `on_missing`
            is `"raise"`.
        ImportError: if a parent package has to be imported to find the module
            (because it is not found statically) and that import fails.
        ValueError: if `fullname` is not a valid absolute module name or
            `on_missing` is not a valid option.

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
    if "" in fullname.split("."):  # empty, relative or with empty components
        raise ValueError(f"Invalid module name: expected an absolute name, got '{fullname}'.")
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

    Regular modules are returned unchanged. Use it wherever module identity
    matters (`is`, `==`, `in`, dict keys, or checks that an object is
    importable under its qualified name), since proxies only compare equal to
    themselves. When replacing an eager import by `lazy_import_module()`,
    check every such use of the imported name.
    """
    if type(module) is _LazyModuleProxy:
        return _LazyModuleProxy._resolve(module)
    return module
