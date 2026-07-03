---
tags: []
---

# [Crash-Consistent Build Caches]

- **Status**: implemented (see *Implementation findings* for refinements)
- **Authors**: TBD (draft prepared for review)
- **Created**: 2026-06-22
- **Updated**: 2026-06-22
- **Related**: [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md),
  [0017 - Toolchain Configuration](0017-Toolchain-Configuration.md),
  [0023 - Fingerprinting](0023-Fingerprinting.md)

In the context of the `gt4py.next` OTF build caches, facing user reports that an
**interrupted** pipeline (Ctrl-C, `SIGKILL`/OOM, node failure, full disk, MPI
rank abort) can leave a cache entry that is **present** (a key lookup reports a
HIT) but **incomplete** (the payload is missing, truncated, or half-written), so
the next run trusts the HIT and **fails on load** instead of rebuilding, we
propose to make both cache layers **crash-consistent** by separating the *commit
point* of an entry from the *work* that produces it: **atomic publish** on the
write side and **validate-on-read + self-heal** on the read side. We considered
relying on the file lock alone, and on DaCe's internal `use_cache` logic, and
reject both because neither survives a writer being killed mid-write.

## Context

### What the fingerprinting refactor did and did not solve

[ADR 0023](0023-Fingerprinting.md) (PR #2648) made cache **keys** correct: a key
is now a deterministic, content-based fingerprint of the step plus its input, so
two runs agree on *which* entry to look up. But a key only answers *"is there an
entry for this input?"* — it says nothing about whether that entry's **payload
is complete and valid**. The reported failures are entirely on this second axis:
the key is fine, the bytes behind it are not.

### The two cache layers

Both backends use the same two independent, fingerprint-keyed layers:

| Layer                  | Where                                                        | Keyed by                                                                      | Stores                                                                                    | Commit semantics today                                                            |
| ---------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Translation cache**  | `CachedStep.persistent` → `FileCache` (`_core/filecache.py`) | structural fingerprint of the ITIR input (`compilable_program_fingerprinter`) | a pickled `ProgramSource` — **self-contained** (gtfn: C++ source; dace: `sdfg.to_json()`) | in-place `pickle.dump`; HIT = `.pkl` file exists                                  |
| **Build-folder cache** | `get_cache_folder()` (`otf/compilation/cache.py`)            | `sha256` of the serialized generated source                                   | the compiled artifacts (`.so`, `.cpp`/SDFG, build metadata)                               | gtfn: `build_data.json` status machine; dace: delegated to DaCe's own `use_cache` |

The two layers are keyed independently (input fingerprint vs. generated-source
hash) and the generated source is a deterministic function of the input, so
there is **no dangling cross-layer reference**: a stale/bad entry in one layer
does not silently point at the other. Each failure mode is therefore local to a
single layer, which keeps the fixes local too.

Crucially, **the translation payload is self-contained** in both backends
(`translation.py` stores `sdfg.to_json(hash=True)` as a string, not a path), so
the only cross-process artifact that can be "present but broken" in the
translation layer is the pickle file itself.

### Key code paths

`FileCache` writes in place and reports HIT on mere existence:

```python
# _core/filecache.py
def __contains__(self, key):  # :54
    return self._get_path(key).exists()  # existence only — not "complete"


def __getitem__(self, key):  # :35
    ...
    return pickle.load(f)  # :40 — raises on a truncated file


def __setitem__(self, key, value):  # :42
    with locking.lock(path := self._get_path(key)):
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=5)  # :46 — NOT atomic vs. a crash
```

`CachedStep` only treats `KeyError` as a miss — any other load exception
propagates and aborts the run:

```python
# otf/workflow.py:333
def __call__(self, inp):
    key = self.cache_key(inp)
    try:
        result = self.cache[key]
    except KeyError:  # EOFError / UnpicklingError NOT caught
        result = self.cache[key] = self.step(inp)
    return result
```

The gtfn build folder *does* have a status marker (`INITIALIZED → CONFIGURED → COMPILED`) — the right idea — but the marker is written non-atomically, the
reader only tolerates a *missing* file (not a corrupt one), and the
"artifact actually exists" check is decoupled from the HIT decision:

```python
# otf/compilation/build_data.py
def read_data(path):  # :76
    try:
        return BuildData.from_json(json.loads((path / _DATAFILE_NAME).read_text()))
    except FileNotFoundError:  # a truncated JSON → JSONDecodeError, UNCAUGHT
        return None


def write_data(data, path):  # :83
    (path / _DATAFILE_NAME).write_text(json.dumps(data.to_json()))  # NOT atomic
```

```python
# otf/compilation/compiler.py:65  (Compiler.__call__)
with locking.lock(src_dir):
    data = build_data.read_data(src_dir)
    if not data or not is_compiled(data) or self.force_recompile:  # :76 — HIT decision
        self.builder_factory(inp, self.cache_lifetime).build()
    new_data = build_data.read_data(src_dir)
    if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
        raise CompilationError(...)  # :82 — verify AFTER deciding not to rebuild
```

The DaCe build folder is weaker still: gt4py sets `compiler.use_cache=True`
(`dace/workflow/common.py:57`), points `sdfg.build_folder` at the keyed cache
folder, and calls `sdfg.compile()` (`dace/workflow/compilation.py:172-174`).
With `use_cache=True`, DaCe's HIT decision is **existence-only**, then it
immediately `dlopen`s:

```python
# dace/sdfg/sdfg.py:2567  (SDFG.compile, installed DaCe)
if not self._recompile or Config.get_bool("compiler", "use_cache"):
    lib_path = compiler.get_binary_name(build_folder, self.name, folder_mode)
    if lib_path.is_file():  # ONLY checks the .so exists
        return compiler.load_precompiled_sdfg(folder=build_folder, sdfg=self)  # dlopen
```

The `.so` is written non-atomically by CMake into the build folder
(`configure_and_compile`), so a kill mid-link leaves a partial `.so` that
`is_file()` happily accepts. gt4py adds **no** build-folder verification around
the DaCe path — there is no equivalent of gtfn's `module_exists`.

## Failure-case taxonomy (part a)

Each row: the interruption window, what is left on disk, what the next run does
today, and a **test recipe** (all are unit-testable by writing a partial state
and asserting a clean rebuild instead of an exception).

### F1 — Truncated translation pickle (gtfn **and** dace)

- **Window**: killed during `pickle.dump` (`filecache.py:46`).
- **On disk**: `…/translation_cache/<hash>.pkl` exists, truncated/empty.
- **Today**: `__contains__` → True; `__getitem__` → `EOFError` /
  `pickle.UnpicklingError`; `CachedStep.__call__` only catches `KeyError`, so it
  **propagates and aborts**. No self-heal.
- **Test**: write a valid entry, truncate the `.pkl` to N bytes, call the cached
  step again, assert it recomputes (and the run succeeds) rather than raising.

### F2 — Truncated `build_data.json` (gtfn)

- **Window**: killed during `write_text` in `write_data`/`update_status`
  (`build_data.py:84`), at any of the three status transitions.
- **On disk**: `gt4py.json` present but not valid JSON.
- **Today**: `read_data` catches only `FileNotFoundError`; `json.loads` raises
  `JSONDecodeError`, **uncaught** → aborts.
- **Test**: build once, overwrite `gt4py.json` with `"{ truncated"`, invoke the
  compiler, assert clean rebuild.

### F3 — gtfn build folder: status `COMPILED` but artifact missing/partial

- **Window**: a complete build whose `.so` is later removed or truncated
  (scratch cleanup, partial `rm`, disk pressure), or a `.so` that exists but is
  not loadable.
- **On disk**: `gt4py.json` = `COMPILED`, module file missing or a truncated
  `.so`.
- **Today**: the HIT gate (`compiler.py:76`) sees `is_compiled(data)` → True and
  **skips the rebuild**; then either `module_exists` is False →
  `CompilationError` (`compiler.py:82`), or the file exists-but-truncated →
  `import_from_path` raises `ModuleNotFoundError` (`importer.py:40`). **No
  self-heal** — `module_exists` only checks `.exists()`, never loadability, and
  it is evaluated *after* the decision not to rebuild.
- **Test**: build once; `unlink` the module file (keep `gt4py.json` =
  `COMPILED`); invoke the compiler; assert clean rebuild. Repeat truncating the
  `.so` to 16 bytes. (Current tests only cover the case where status is manually
  downgraded to `CONFIGURED` — `test_compiledb.py::test_compiledb_project_is_relocatable`
  — i.e. the *self-healing* case, never the stuck-`COMPILED` case.)

### F4 — dace build folder: partial `.so` accepted by DaCe `use_cache`

- **Window**: killed during `sdfg.compile()` (`compilation.py:174`) after the
  `.so` path appears but before linking finishes.
- **On disk**: `build/<name>.so` (or `lib…`) exists, truncated/partial.
- **Today**: next `sdfg.compile()` → `lib_path.is_file()` True
  (`sdfg.py:2572`) → the compile step silently accepts the truncated library
  (since #2587 it never loads it: `return_program_handle=False`); every later
  `DaCeCompilationArtifact.load()` raises `RuntimeError` (`file format not recognized`) from the failed `dlopen`. No self-heal. (A *missing* library, by
  contrast, already self-heals — see *Implementation findings*.)
- **Test**: compile once; truncate the produced `.so` and remove the completion
  marker (mimicking a kill mid-link); run the dace compilation step again;
  assert clean rebuild (this needs a real compiler, so it is an
  integration-level test gated like the other dace build tests).

### F5 — Partial source / config state (both, **already handled**)

- gtfn: interruption before `CONFIGURED`/`COMPILED` leaves status `INITIALIZED`
  / `CONFIGURED`; the status machine (`compiledb.py:121-133`) resumes correctly,
  and CMake/Ninja incrementality covers the in-`build/` state. Documented here
  to scope it out — F5 is **not** a defect, and the fix must preserve this
  resume behavior.

**Concurrency is orthogonal.** `locking.lock` (`_core/locking.py`) correctly
serializes concurrent builders of the same key (MPI ranks, xdist workers) but
gives **zero** crash consistency: a killed lock holder releases the lock and
leaves its partial entry behind. F1–F4 all occur with the lock working as
intended.

## How other frameworks handle this (part b)

Every mature compiler/JIT cache converges on the **same two-layer defense**, and
it maps directly onto F1–F4. Primary sources surveyed: CPython `.pyc`, OpenAI
Triton, PyTorch TorchInductor, Numba, ccache, sccache, Bazel disk cache; JAX and
Halide as instructive negatives.

1. **Write side — atomic publish.** Never write into the final path. Write to a
   temp file/dir *on the same filesystem*, then `os.replace()` (single file) or
   rename the whole dir into place. `rename(2)` atomically swaps the directory
   entry, so a reader — including the next run after a kill — sees old-or-new,
   never torn. Used by **CPython** (`_write_atomic`, temp `.{id}` →
   `os.replace`), **Triton** (`FileCacheManager.put`, temp dir → `os.replace`,
   with the comment *"filepath cannot see a partial write"*), **PyTorch**
   (`write_atomic`, `.{pid}.{tid}.tmp` → `rename`), **Numba**
   (`IndexDataCacheFile`, `.tmp.{uuid4}` → `os.replace`), **ccache**
   (`AtomicFile`, temp in dest dir → `rename`), **sccache**, **Bazel** (UUID
   temp → `renameToleratingConcurrentCreation`). The temp **must** be a sibling
   of the target (cross-filesystem rename fails with `EXDEV`).

2. **Read side — validate & self-heal.** Treat *any* load failure (missing,
   truncated, `EOFError`, `UnpicklingError`, bad magic/version, stamp/checksum
   mismatch) as a **MISS that rebuilds**, never an exception that aborts;
   optionally evict the bad entry. **CPython** validates a magic + flags +
   (mtime/size | source-hash) header and silently recompiles on any mismatch
   (PEP 552). **ccache** stores an XXH3-128 checksum and rebuilds on mismatch.
   **PyTorch** wraps load in a bare `except Exception` → skip. **Numba** rebuilds
   on version/stamp mismatch. This is the layer that catches the residual cases
   atomic rename cannot (power-loss torn writes, bit rot, format skew across tool
   upgrades).

3. **"Done"/marker written last** is the practical way to make a *multi-file
   directory* entry atomic without renaming the whole dir: build payloads in
   place, then publish a single small marker atomically and **last**; readers
   check the **marker, not the payload**. **Numba**'s index file is exactly this;
   gt4py's `build_data.status == COMPILED` is an *attempt* at it, broken by F2/F3.

4. **Lock ≠ crash consistency.** A lock serializes live builders; an atomic
   rename gives crash safety. They are orthogonal and you want both. **JAX**'s
   current cache is the cautionary tale: it takes a `filelock` but then
   `write_bytes` directly to the final path, so a kill mid-write leaves a
   truncated entry under a valid key — *exactly the reported gt4py bug*.

5. **fsync is for power loss, not process kill.** Atomic rename alone survives
   Ctrl-C/`SIGKILL`/OOM (the reported failures). Surviving a *power cut* also
   needs `fsync(data)` *before* the rename and `fsync(dir)` after. Almost no JIT
   cache bothers (CPython calls its helper *"best-effort"*; Triton, PyTorch,
   Numba, ccache, sccache all skip it); **Bazel** is the lone fsync-er. The
   field consensus — and our recommendation — is temp+`os.replace` **without**
   fsync, backed by validate-on-read for the rare power-loss case, because
   compiled artifacts are reproducible and fsync-per-write is real latency
   (especially on Lustre/NFS scratch).

Full scorecard, code sketches, and per-pattern coverage/non-coverage are in the
research appendix kept with this proposal; the citations are inlined above.

## Decision (proposed)

Adopt the two-layer defense, implemented as a **small shared helper** plus
**local edits** at each cache site. Principle: *a key is a HIT only when the
payload behind it is proven complete; anything else is a clean MISS that
rebuilds.*

### 0. Shared helper — `gt4py._core/cache_utils.py` (new)

```python
def atomic_write_bytes(target: pathlib.Path, data: bytes) -> None:
    """Write `data` to `target` atomically: temp sibling + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, target)  # atomic on POSIX & Windows
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise
```

(`atomic_write_text` wraps it.) No fsync by default — see consequence below.

### 1. Translation cache — `FileCache` (fixes F1)

- **Write atomically**: route `__setitem__` through `atomic_write_bytes`
  (pickle to `bytes`, then atomic publish). Keep the existing `locking.lock` for
  concurrency.
- **Tolerant read**: in `__getitem__`, catch `(EOFError, pickle.UnpicklingError, OSError, AttributeError, …)`, **evict** the bad file, and raise `KeyError` so
  the existing `CachedStep.__call__` miss-path recomputes it. This makes a
  corrupt entry self-heal *without* touching `CachedStep`.

```python
def __getitem__(self, key):
    if key not in self:
        raise KeyError(key)
    with locking.lock(path := self._get_path(key)):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError, AttributeError, ImportError) as e:
            path.unlink(missing_ok=True)  # self-heal: drop the unusable entry
            raise KeyError(key) from e
```

### 2. gtfn build folder — `build_data` + `Compiler` (fixes F2, F3)

- **Atomic marker write**: `write_data`/`update_status` use
  `atomic_write_text`. The `COMPILED` status becomes a proper commit marker
  written last (it already is — `_run_build` sets it after the link).
- **Tolerant marker read**: `read_data` catches `json.JSONDecodeError` (and
  `OSError`) in addition to `FileNotFoundError`, returning `None` (→ rebuild).
- **Fold verification into the HIT decision (self-heal)**: rebuild unless the
  marker says `COMPILED` *and* the module is present. Replace the two-step
  "decide, then verify-and-raise" with one predicate:

```python
def _is_usable(data, src_dir) -> bool:
    return bool(data) and is_compiled(data) and module_exists(data, src_dir)


with locking.lock(src_dir):
    if self.force_recompile or not _is_usable(build_data.read_data(src_dir), src_dir):
        self.builder_factory(inp, self.cache_lifetime).build()
    new_data = build_data.read_data(src_dir)
    if not _is_usable(new_data, src_dir):
        raise CompilationError(...)  # genuine build failure, not a stale cache
```

A stale `COMPILED`-but-missing folder now rebuilds instead of raising. (Optional
hardening: on the final `import_from_path`, treat an `ImportError` as one-shot
self-heal — wipe + rebuild once — to also cover the truncated-but-present `.so`.)

### 3. dace build folder — completion marker written last (fixes F4)

gt4py cannot change DaCe's existence-only gate, but it owns the wrapper around
`sdfg.compile()`. A *missing* library already self-heals (`is_file()` is False
and `SDFG.regenerate_code` defaults to `True`, so DaCe regenerates), so the only
crash-inconsistent state is a **truncated-but-present** library that the
`use_cache` gate accepts.

Since #2587 split compilation from loading, the compile step calls
`sdfg.compile(return_program_handle=False)` — **no `dlopen` happens at compile
time**, and the later `DaCeCompilationArtifact.load()` (possibly in another
process) has neither the build lock nor the dace config context needed to
recompile. Self-heal-on-load-failure is therefore no longer viable; the fix is
the commit-marker pattern, applied inside the locked compile step:

- After a completed `sdfg.compile()`, touch a `.gt4py_compile_complete` marker
  in the build folder (**written last**; an interrupted `touch` just means no
  marker).
- Before compiling, if the marker is **absent** the previous build was
  interrupted: delete the cached `lib<name>.*` / `libdacestub_<name>.*` so
  DaCe's `is_file()` gate misses and it rebuilds. The marker is also removed
  before every compile, so an interrupted *re*build (e.g. after external
  cleanup deleted the library) cannot leave a stale marker next to a truncated
  library.

Deleting only the library files (rather than `shutil.rmtree` of the whole
folder) keeps the `locking.lock` file — which lives *inside* the build folder —
intact while the lock is held.

### Alternatives considered

- **Whole-directory atomic publish (Pattern 2)** — build into a sibling temp dir
  and `os.rename` the directory into the keyed path; the strongest fix (an entry
  is *all-or-nothing*, no marker needed). **Rejected as the primary** because the
  in-place build systems fight it: gtfn's compiledb relies on CMake/Ninja
  incrementality and relocation, and DaCe writes directly into `sdfg.build_folder`
  and returns a `CompiledSDFG` bound to that path — renaming after compile means
  rebinding the handle's `build_folder`, and a publish-once dir loses the F5
  resume behavior. Kept as a documented option if marker+self-heal proves
  insufficient.
- **Lock-only / status-quo** — rejected: F1–F4 all reproduce with the lock
  working (see "Concurrency is orthogonal").
- **fsync on every write** — rejected as default: protects only against power
  loss (not the reported process-kill cases), costs real latency on Lustre/NFS,
  and validate-on-read already absorbs the rare torn write. Leave a config knob
  if a power-loss corruption is ever observed in practice.
- **Trust DaCe `use_cache`** — rejected: its HIT gate is `is_file()` only
  (`sdfg.py:2572`), which is the root of F4.

## Consequences

What becomes easier / safer:

- An interrupted run never poisons the cache: the next run cleanly rebuilds the
  affected entry. F1–F4 turn from hard aborts into transparent rebuilds.
- One shared atomic-write helper is reused by both cache layers and both
  backends; the gtfn and dace build folders become conceptually symmetric
  (commit marker written last, verified before reuse).
- The read-side tolerance also absorbs **format/version skew**: a cache written
  by an older gt4py/CPython that no longer unpickles becomes a clean miss rather
  than a crash (complements `BUILD_CACHE_VERSION_ID` from ADR 0023).

Trade-offs / what to watch:

- **Not power-loss-durable by default** (no fsync). Acceptable: artifacts are
  reproducible, and validate-on-read catches a torn write on the *next* read. A
  hard power cut may still require regenerating the last in-flight entry.
- **Broad `except` on read** must be scoped so it cannot mask a real bug as an
  infinite rebuild loop. Corrupt entries are evicted silently: `_core` has no
  established logging, and a per-entry warning would spam N-rank MPI runs; a
  persistent failure still surfaces as a rebuild-then-error, not a loop.
- **Orphan temp files** (`.<name>.*.tmp`, interrupted between write and
  replace) accumulate; a cheap sweep of stale `*.tmp` on cache-dir touch (as
  sccache does on `init`) keeps them bounded. Low priority.
- **NFS/Lustre**: `os.replace` atomicity holds for same-directory renames on
  POSIX filesystems; locks degrade worse than atomic-rename there, so the
  rename+validate combo is the more robust choice for HPC scratch — but validate
  carefully on the target filesystem.

## Implementation plan

1. `gt4py._core/cache_utils.py`: `atomic_write_bytes` / `atomic_write_text`
   (+ unit tests, incl. a "killed before replace leaves target untouched" test
   via an injected failure between write and `os.replace`).
2. `FileCache`: atomic `__setitem__`, tolerant self-healing `__getitem__`
   (§1) — fixes F1 for **both** backends.
3. `build_data`: atomic writes + broaden `read_data` (§2) — fixes F2.
4. `Compiler.__call__`: single `_is_usable` predicate with self-heal (§2) —
   fixes F3.
5. dace `compilation.py`: completion marker written last + pre-clean around
   `sdfg.compile()` (§3) — fixes F4.

Each step is independently shippable; (2) alone already removes the most-reported
symptom (truncated translation pickle).

## Test plan

`tests/.../compilation_tests/build_systems_tests/test_cache_consistency.py`
(gtfn, F2/F3) and `.../dace_tests/test_dace_cache_consistency.py` (F4), plus
F1 in `core_tests/.../test_filecache.py` and
`next_tests/.../otf_tests/test_workflow.py`. For each failure mode: **build a
valid entry, corrupt it to mimic an interruption, assert clean recovery**.

- F1: `c[k] = v`; truncate the `.pkl`; assert reads raise `KeyError` (miss),
  the stale file is evicted, and `CachedStep.persistent` recomputes.
- F2: build; overwrite `gt4py.json` with invalid JSON; assert the compiler
  rebuilds to `COMPILED` with the module present.
- F3: build; `unlink` the module while leaving status `COMPILED`; assert rebuild
  rather than `CompilationError`.
- F4: compile a real CPU SDFG; truncate `lib<name>.so` and remove the completion
  marker (mimicking a kill mid-link); assert the compile step rebuilds and the
  resulting library `dlopen`s cleanly. (Since #2587 the compile step never loads
  the library in-process, so no `dlopen`-handle gymnastics are needed.)
- Regression for F5: the existing `INITIALIZED`/`CONFIGURED` resume tests still
  pass unchanged.

## Implementation findings

Discovered while writing the tests (they refine, and in one place simplify, the
Decision above):

1. **A truncated shared library cannot be reproduced in-process by
   compile-then-truncate**: glibc caches the `dlopen` handle, so the truncated
   file is never re-read. The failure only manifests on a fresh *re-run* process
   (the user's actual scenario). The gtfn truncated-module case (a C-extension
   `.so`, which Python cannot cleanly unload in-process) is **not** unit-tested —
   the realistic, in-process **F3-missing** case is covered instead. The dace
   test is unaffected: since #2587 the compile step never loads the library
   in-process.
2. **A missing dace library self-heals without help.** `SDFG.regenerate_code`
   defaults to `True` (`dace/sdfg/sdfg.py:550`), so a *missing* library always
   regenerates; only the truncated-but-present library is a bug.
3. **#2587 (compile/load split) forced the marker design.** An earlier revision
   of this fix self-healed on load failure inside the combined compile+load
   step. With `sdfg.compile(return_program_handle=False)` no load happens at
   compile time, and `DaCeCompilationArtifact.load()` cannot recompile (no
   build lock, no dace config context) — so §3 uses the commit-marker pattern
   originally proposed here.

## Resolved decisions

1. **Marker vs. whole-dir rename** → marker for dace (`.gt4py_compile_complete`,
   written last; forced by the #2587 compile/load split, finding #3). gtfn keeps
   its `COMPILED` status as the commit marker (now atomic + tolerant). Whole-dir
   rename remains a documented future option.
2. **Location of `atomic_write_*`** → `gt4py._core/cache_utils.py`, alongside
   `FileCache`/`locking`.
3. **fsync** → not implemented (per maintainer); atomic rename covers
   process-kill, validate-on-read covers the rare power-loss torn write. A
   `GT4PY_BUILD_CACHE_FSYNC` knob can be added if ever needed.
4. **Orphan-temp sweep** → deferred (low priority).
5. **gtfn truncated-but-present module** → documented gap: the `is_usable` gate
   covers the missing-artifact case; a present-but-unloadable `.so` (rare, needs
   post-build corruption) is not self-healed, keeping the fix simple.

## References

- `src/gt4py/_core/filecache.py`, `src/gt4py/_core/locking.py`
- `src/gt4py/next/otf/workflow.py` — `CachedStep`
- `src/gt4py/next/otf/compilation/{cache.py,compiler.py,build_data.py,importer.py}`
- `src/gt4py/next/otf/compilation/build_systems/{compiledb.py,cmake.py}`
- `src/gt4py/next/program_processors/runners/gtfn.py` — `cached_translation` trait
- `src/gt4py/next/program_processors/runners/dace/workflow/{common.py,translation.py,compilation.py,factory.py}`
- Installed DaCe: `dace/sdfg/sdfg.py::SDFG.compile`, `dace/codegen/compiler.py::configure_and_compile`
- External patterns (primary sources): CPython `importlib/_bootstrap_external.py::_write_atomic` + PEP 552; Triton `runtime/cache.py`; PyTorch `_inductor/codecache.py::write_atomic`; Numba `core/caching.py`; ccache `core/atomicfile.cpp` + `core/cacheentry.cpp`; sccache `lru_disk_cache`; Bazel `remote/disk/DiskCacheClient.java`; LWN "Ensuring data reaches disk" (https://lwn.net/Articles/457667/); `rename(2)` (https://man7.org/linux/man-pages/man2/rename.2.html); `os.replace` (https://docs.python.org/3/library/os.html#os.replace)
