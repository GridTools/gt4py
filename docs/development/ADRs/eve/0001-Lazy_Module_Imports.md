---
tags: [eve, utils, imports, dependencies]
---

# Lazy Module Imports

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-11-20
- **Updated**: 2026-11-20

In the context of importing optional and heavy external libraries (e.g.
`cupy`, `cupyx.profiler`, `jax.numpy`, `dace` subpackages) only when they are
actually used, facing the fact that the established lazy-loading approach
(`lazy_loader.load()` and the standard library `importlib.util.LazyLoader`)
breaks for subpackages and leaks import errors of broken installations into
unrelated code, we decided to implement `gt4py.eve.utils.lazy_import_module()`
as a forwarding proxy which is not registered in `sys.modules` and resolves
itself with a plain `importlib.import_module()` call on first attribute
access, to achieve exactly the standard import semantics with public APIs
only. We considered registering lazy module objects in `sys.modules` and
accept that proxies are not identical to the real modules.

## Context

The usual pattern for optional libraries in GT4Py is:

```python
try:
    import cupy as cp
except ImportError:
    cp = None
```

which imports the library eagerly at GT4Py import time, even if it is never
used. The `lazy_loader.load()` function of the scientific-python
[lazy-loader](https://github.com/scientific-python/lazy-loader) project
creates a module object with `importlib.util.LazyLoader`, registers it in
`sys.modules` and executes the module code on first attribute access. Its own
documentation warns against using it for subpackages, and our evaluation
found that this design has fundamental problems in our context:

- `importlib.util.find_spec("a.b")` imports `a` eagerly.
- Even when the spec of `a.b` is found without executing `a`, touching the
  lazy `a.b` first executes `b` before `a`, which breaks the common layout of
  a parent re-exporting names from a child that imports the parent (verified
  on CPython 3.12–3.14). The standard library `_LazyModule` cannot be
  subclassed to fix this, since it checks `type(self) is _LazyModule`.
- A custom "parent-first" lazy module class registered in `sys.modules` can
  fix the ordering, but an adversarial review found it needs several patches
  over CPython internals: per-module import locks
  (`importlib._bootstrap._ModuleLockManager`) to avoid deadlocks with the
  import system, deferring the attachment of children to avoid skipping their
  import-time side effects, answering some dunder attributes without loading,
  and a clean rollback on failures.
- Two problems are unfixable while the lazy module is in `sys.modules`:
  - Any scan of `sys.modules` loads it. `inspect.getmodule()` does this, and
    GT4Py itself calls it at import time (via `inspect.getframeinfo()` in
    `eve.codegen` templates), so importing `gt4py.next` loaded every lazy
    module.
  - An installed-but-broken library (e.g. `cupy` without a working CUDA
    runtime) raises its `ImportError` inside unrelated `sys.modules` probes,
    e.g. `array_api_compat.array_namespace()` or `inspect.stack()`.

## Decision

`lazy_import_module(fullname, *, on_missing="raise" | "none")`, defined in
`gt4py.eve.utils.lazy_imports` and re-exported from `gt4py.eve.utils`:

1. Returns the real module if it is already in `sys.modules`.
2. Checks whether the module is installed without executing any module code,
   walking the parent chain with the finders of `sys.meta_path` (using the
   runtime `__path__` of imported parents and the spec search locations of
   the others). If a child is not found below a parent which is not imported
   yet, the parent is imported for real and the lookup is retried, since its
   runtime `__path__` can differ from the static one. `None` entries in
   `sys.modules` and non-package parents count as missing, like with plain
   imports. Missing modules either raise `ModuleNotFoundError` or return
   `None`, depending on `on_missing`.
3. Otherwise returns a `types.ModuleType` subclass instance which is **not**
   registered in `sys.modules`. Any attribute access except `__name__` and
   `__class__` calls `importlib.import_module(fullname)` and forwards to the
   real module. The real module is cached only once it is fully initialized
   and present in `sys.modules`, so failed imports are retried as with plain
   imports. Import errors during dunder attribute probes are converted to
   `AttributeError` so introspection of namespaces holding a proxy to a
   broken library does not crash.

`resolve_lazy_module()` returns the real module of a proxy and
`is_lazy_module()` identifies proxies.

We chose this option because import order, locking, import-time side
effects, runtime `__path__` changes, self-replacing modules and failure
rollback are all handled by the standard import system, with no dependency on
CPython internals (except a defensive read of the undocumented
`ModuleSpec._initializing` flag), and because broken libraries stay invisible
to code which does not use them.

## Consequences

- Optional dependencies can be declared at module level without paying their
  import cost until they are used.

- A proxy is not the real module object: `proxy is module`, `proxy == module`
  and dict lookups keyed by the real module fail. Proxies keep the default
  identity-based equality and hash; delegating them to the real module was
  rejected because the hash of a proxy could change after a transient import
  failure, breaking the hash/equality contract. Code relying on module
  identity must call `resolve_lazy_module()`. When migrating an existing
  import to `lazy_import_module()`, check every `is`, `==`, `in` and
  dict-key use of the name. Known sites at the time of writing:

  - `_core/ndarray_utils.py` (device translators registry keyed by module),
  - `next/field_utils.py` (`array_ns == cp`),
  - `next/embedded/array_ops.py` (`_ops_by_namespace` registry),
  - `next.fingerprinting.strict_fingerprinter`, which rejects objects that
    are not importable under their qualified name (a deconstructor resolving
    proxies needs to be registered in `gt4py.next`).

  Registering the resolved module at import time imports the library, so
  register lazily or key by module name where laziness matters.

- Each attribute access goes through the proxy (~70 ns overhead after
  loading); hot code should bind attributes or the resolved module locally.

- `lazy_import_module("a")` does not make `a.b` reachable unless `a` imports
  it itself (same as `import a`): each used subpackage needs its own call.

- `on_missing` only covers libraries which are not installed. Errors of
  installed-but-broken libraries surface on first access, or at call time if
  the fallback path needs to import a broken parent package.

- Probing dunder attributes (e.g. `hasattr(proxy, "__wrapped__")`), `dir()`
  and `vars()` import the module. `importlib.reload()` needs the resolved
  module.

- For type checkers, the call must be placed in the `else` branch of an
  `if TYPE_CHECKING:` block containing a regular import, which hides the
  `None` case from them.

## Alternatives considered

### `lazy_loader.load()` / `importlib.util.LazyLoader`

- Good, because the lazy module is the real module object (identity is
  preserved) and the implementation is small.
- Bad, because subpackages either import their parents eagerly or are
  executed before their parents, breaking common package layouts.

### Custom parent-first lazy module registered in `sys.modules`

- Good, because identity is preserved and subpackages are supported.
- Bad, because it reimplements CPython internals (private per-module import
  locks, dict-diff reapplication, rollback), adds deadlock scenarios compared
  to plain imports on CPython 3.12, and cannot prevent `sys.modules` scans
  (`inspect.getmodule()`) from loading the module or broken libraries from
  breaking unrelated code.

### PEP 810 explicit lazy imports

PEP 810 (status: Final) adds `lazy import` statements in Python 3.15, which
would supersede this utility. It is not available in the supported Python
versions (3.12–3.14).

## References

- [scientific-python/lazy-loader](https://github.com/scientific-python/lazy-loader)
- [`importlib.util.LazyLoader`](https://docs.python.org/3/library/importlib.html#importlib.util.LazyLoader)
- [PEP 562 – Module `__getattr__` and `__dir__`](https://peps.python.org/pep-0562/)
- [PEP 810 – Explicit lazy imports](https://peps.python.org/pep-0810/)
