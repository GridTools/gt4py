---
tags: []
---

# [Crash-Consistent Build Caches]

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-06-22
- **Updated**: 2026-07-03

In the context of the `gt4py.next` OTF build caches, facing user reports that
an interrupted pipeline (Ctrl-C, `SIGKILL`/OOM, full disk) can leave a cache
entry that a key lookup reports as a HIT but whose payload is truncated or
missing — so the next run fails on load instead of rebuilding — we decided to
make every cache write an **atomic publish** (temp sibling + `os.replace`) and
every cache read **validate + self-heal** (any load failure → treat as miss →
rebuild). We considered relying on the file lock, `fsync`-ing writes, and
whole-directory renames, and accept that a hard power loss may still require
regenerating the last in-flight entry.

## Context

[ADR 0023](0023-Fingerprinting.md) made cache *keys* correct, but a key only
answers "is there an entry for this input?" — not whether the entry's payload
is complete. Both backends use two independent fingerprint-keyed layers, each
with its own present-but-broken states:

| Layer                           | Broken state after an interruption                       | Old behavior on re-run                                                               |
| ------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| translation cache (`FileCache`) | truncated pickle                                         | `EOFError`/`UnpicklingError` aborts                                                  |
| gtfn build folder               | truncated `gt4py.json`                                   | `JSONDecodeError` aborts                                                             |
| gtfn build folder               | status `COMPILED`, module deleted (e.g. scratch cleanup) | `CompilationError`                                                                   |
| dace build folder               | truncated `lib<name>.so` (kill mid-link)                 | accepted by dace's existence-only `use_cache` gate; `dlopen` fails on every `load()` |
| compiledb template (shared)     | truncated `compile_commands.json`                        | `JSONDecodeError` for *every* program with that configuration                        |

(A build interrupted *before* reaching `COMPILED` already resumes correctly via
the gtfn status machine and must keep doing so.)

The file lock is orthogonal: it serializes concurrent builders but a killed
lock holder releases the lock and leaves its partial entry behind. Mature
compiler caches (CPython `.pyc`, Triton, TorchInductor, Numba, ccache, sccache)
all converge on the same write-atomically / validate-on-read combination used
here.

## Decision

A shared helper `gt4py._core.cache_utils.atomic_write_bytes/_text` writes to a
uniquely-named sibling temp file and `os.replace`s it into place, so a reader
sees old-or-new content, never a torn file. The temp file is created with a
plain `open` so the result keeps umask-derived permissions. No `fsync`: it only
adds power-loss durability, costs real latency on parallel filesystems, and
validate-on-read absorbs the rare torn write.

Per layer:

- **Translation cache** — `FileCache.__setitem__` publishes atomically;
  `__getitem__` treats any unpicklable entry (`EOFError`, `UnpicklingError`,
  `OSError`, and `AttributeError`/`ImportError` from version skew) as a miss,
  evicting the file and raising `KeyError` so `CachedStep` recomputes.
- **gtfn build folder** — `build_data.write_data` publishes atomically, so the
  `COMPILED` status is a proper commit marker written last; `read_data` returns
  `None` for unreadable metadata. The cache-HIT gate is a single predicate
  `is_usable(data, src_dir)` = status `COMPILED` *and* module file present;
  anything else rebuilds (preserving the partial-build resume states).
- **compiledb template** — published atomically; `_cc_find_compiledb`
  validates the JSON on HIT and evicts an unreadable template so it is
  regenerated under the existing lock.
- **dace build folder** — dace's `use_cache` gate accepts a library on
  mere existence, and the compile step never loads it
  (`sdfg.compile(return_program_handle=False)` since #2587), so a truncated
  library survives until `DaCeCompilationArtifact.load()` — which has neither
  the build lock nor the dace config context to recompile. Instead, the locked
  compile step writes a `.gt4py_compile_complete` marker **after** each
  completed compile (and removes it before compiling); if the marker is absent,
  the previous build was interrupted and the stale `lib<name>.*` /
  `libdacestub_<name>.*` files are deleted so dace rebuilds them. A *missing*
  library needs no handling: `SDFG.regenerate_code` defaults to `True`, so dace
  regenerates it anyway.

## Alternatives considered

- **Lock-only / status quo**: gives no crash consistency (see Context).
- **Whole-directory atomic publish**: strongest guarantee, but fights the
  in-place CMake/Ninja incremental builds, dace's `build_folder` binding, and
  the partial-build resume behavior. Remains an option if marker + self-heal
  proves insufficient.
- **`fsync` on every write**: protects only against power loss, not the
  reported process-kill cases; rejected as default (a knob can be added if
  power-loss corruption is ever observed).
- **Self-heal on load failure for dace**: worked before #2587, but the
  compile/load split moved loading out of the step that can rebuild.

## Consequences

- An interrupted run no longer poisons the cache; the affected entry is rebuilt
  transparently on the next run. Version-skew pickles also become clean misses.
- Corrupt entries are evicted silently (`_core` has no logging infrastructure,
  and per-entry warnings would spam multi-rank runs); a persistent failure
  still surfaces as rebuild-then-error, not a loop.
- Known gaps: a gtfn module that is present but truncated *after* a completed
  build (torn write at power loss) is not self-healed; orphaned `.tmp` files
  from kills between write and rename are not swept.
