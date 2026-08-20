---
name: translation-cache
description: Clear and verify the gt4py.next translation cache before measuring a change to SDFG or code generation. TRIGGER before benchmarking, profiling, or judging any edit to a dace transformation, an optimization pass, gt_auto_optimize, gtfn codegen, or anything else affecting how a program is translated — and whenever a change appears to have "no effect" on generated code. SKIP for changes to the program itself (icon4py source, static args, domain sizes), which invalidate the cache on their own.
---

# translation-cache

gt4py.next caches the *translation* step — the optimized SDFG for dace, the
generated source for gtfn — keyed by the program and `gt4py.__version__`, not by
the gt4py sources. Edit a transformation or an optimization pass and that key is
unchanged: the next run replays the cached translation and the pass never runs,
while the build step still recompiles from it. Benchmark without clearing the
cache and you are measuring the old compiler.

For an editable install a *commit* changes the version and invalidates the cache.
Uncommitted edits do not — the dirty marker is a constant suffix — and a
non-editable install never does.

## Recipe

Run in the environment and working directory of the run being measured:

```bash
gt4py-next-cache delete --program '<name glob>' --yes
gt4py-next-cache list --filter '<name glob>' --fail-if-cached
```

`delete` reports what it removed and exits non-zero if the selector matched
nothing. The `list` call is the gate — non-zero while any entry for that glob is
still there — so require it to pass **before** launching an expensive job.

Read the output in the one direction that holds: **no entries for a program**
means it will be re-translated; **entries present** is a reason to delete, never
evidence that a run replayed. Deleting an entry that would not have been hit
costs nothing.

When the cache is out of reach (a compute node, a container), or to invalidate
everything at once, salt both caches instead:

```bash
export GT4PY_BUILD_CACHE_VERSION_ID=$(git -C <gt4py checkout> rev-parse HEAD)
```

Use a nonce rather than the commit hash when iterating on uncommitted changes.

## The two caches

Run `gt4py-next-cache path` for where they are in the current environment —
worth checking first, because under the default session lifetime the cache sits
in a temporary directory that is deleted when the process exits, so nothing lands
in `.gt4py_cache` unless `GT4PY_BUILD_CACHE_LIFETIME=persistent` is set.

The build cache and the translation cache are separate and hit independently, so
clearing only the build folder is **not enough**: the build step recompiles from
the replayed translation, and the library is rebuilt while every transformation
is skipped. `gt4py-next-cache list --by-program` shows both side by side, and
flags the combination that hides this — a cached translation with no usable build
folder.

`gt4py-next-cache --help` covers the remaining flags; the same tool runs as
`python -m gt4py.next.gt_cache_manager` when the console script is not on `PATH`.

## Anti-pattern

**A fresh library mtime proves recompilation, not re-translation.** Never cite it
as evidence that a pass ran. Two further tells that were misread once and cost
four multi-node benchmark jobs:

- Successive SDFG dumps differing only in `guid` fields — the signature of
  unpickling the same cached object twice, not of a regenerated SDFG.
- Debug logging added inside a pass producing no output — the pass was never
  called, rather than never applicable.

To prove a pass executed, assert on something it changes, or log inside it and
confirm the log appears.

## Related

- `scripts/python/dace_determinism.py` answers the neighbouring question: whether
  dace codegen is deterministic across two runs.
