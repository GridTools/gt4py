---
name: translation-cache
description: Clear and verify the gt4py.next translation cache before measuring a change to SDFG or code generation. TRIGGER before benchmarking, profiling, or judging any edit to a dace transformation, an optimization pass, gt_auto_optimize, gtfn codegen, or anything else affecting how a program is translated — and whenever a change appears to have "no effect" on generated code. SKIP for changes to the program itself (icon4py source, static args, domain sizes), which invalidate the cache on their own.
---

# translation-cache

gt4py.next replays a cached translation unless the *program* changed. Changing
gt4py does not count. Benchmark without clearing it and you are measuring the old
compiler.

## The two caches

Both live side by side under the cache base (`<cwd>/.gt4py_cache` by default):

| Cache             | Where                                             | Holds                                    |
| ----------------- | ------------------------------------------------- | ---------------------------------------- |
| Build cache       | `<program>_pyext_<hash>_<version>/`               | build artifacts, the compiled library    |
| Translation cache | `translation_cache/` (dace), `gtfn_cache/` (gtfn) | the **already optimized** SDFG / sources |

Clearing only the `*_pyext_*` folder is **not enough**: the build step recompiles
from the replayed translation, so the library is rebuilt while every
transformation and optimization pass is skipped.

The key is a fingerprint of the program plus `gt4py.__version__`. For an editable
install a *commit* changes that version and invalidates the cache — uncommitted
edits do not, and a non-editable install never does.

## Recipe

Run in the environment and working directory of the run being measured:

```bash
gt4py-next-cache delete --program '<name glob>' --yes
gt4py-next-cache list --filter '<name glob>' --fail-if-cached
```

`delete` reports what it removed and exits non-zero if the selector matched
nothing. The `list` call is the gate: it exits non-zero while any entry for that
glob is still there, so a job script can require an empty cache before launching.
Drop the filter to cover every program, and add `--include-build-dirs` to the
delete to force a full rebuild too.

Everything the tool reports is a **count of what is on disk, not a prediction**.
Whether an entry is actually hit depends on a fingerprint taken at run time over
the lowered program and its arguments, which the tool cannot compute. So entries
show up that are already unreachable — after the program's source was edited, or
after the gt4py version changed, since entries record neither. Read it in the one
direction that holds: *no entries for a program* means that program will be
re-translated; *entries present* is a reason to delete, never evidence that a run
replayed. Deleting an entry that would not have been hit costs nothing.

`list --by-program` adds one row per program with its build folders, which is
where the two-cache trap becomes visible: a cached translation with no usable
build folder means the next run rebuilds the library while replaying the
translation.

Without `--yes`, `delete` lists what it matched and asks for confirmation on a
terminal; use `--dry-run` to preview and `--yes` in scripts and job files, where
there is no terminal to ask.

`gt4py-next-cache` is installed with gt4py; `python -m gt4py.next.gt_cache_manager`
is the same tool, for when the console script is not on `PATH`.

Worked example — a new dace transformation, measured on one dycore program:

```bash
gt4py-next-cache list --by-program --filter 'apply_divergence_damping*'
# PROGRAM                                 ENTRIES  BUILDS
# apply_divergence_damping_and_update_vn  3 dace   2
gt4py-next-cache delete --program 'apply_divergence_damping*' --yes
gt4py-next-cache list --filter 'apply_divergence_damping*' --fail-if-cached
# -> no entries, exit 0  <- now re-translation is guaranteed
srun ...   # the pass actually runs
```

Alternative, when the cache is out of reach (a compute node, a container) or you
want everything invalidated at once:

```bash
export GT4PY_BUILD_CACHE_VERSION_ID=$(git -C <gt4py checkout> rev-parse HEAD)
```

That salts both caches, so it also forces a rebuild. Use a nonce instead of the
commit hash when iterating on uncommitted changes.

## Anti-pattern

**A fresh library mtime proves recompilation, not re-translation.** Never cite it
as evidence that a pass ran. Two more tells that were misread once and cost four
multi-node benchmark jobs:

- Successive SDFG dumps differing only in `guid` fields — that is the signature of
  unpickling the same cached object twice, not of a regenerated SDFG.
- Debug logging added inside a pass producing no output — the pass was never
  called, rather than never applicable.

To prove a pass executed, assert on something it changes, or log inside it and
confirm the log appears.

## Related

- `gt4py-next-cache path` prints the cache directories this environment uses.
  Worth checking first: with the default session lifetime the cache lives in a
  temporary directory that is deleted when the process exits, so nothing lands in
  `.gt4py_cache` unless `GT4PY_BUILD_CACHE_LIFETIME=persistent` is set.
- `scripts/python/dace_determinism.py` answers the neighbouring question: whether
  dace codegen is deterministic across two runs.
