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
python -m gt4py.next.gt_cache_manager delete --program '<name glob>' --yes
python -m gt4py.next.gt_cache_manager status --program '<name glob>' --fail-if-cached
```

The `status` call is the gate: it exits non-zero while any matched program would
still replay. Require it to pass **before** launching an expensive job. Drop
`--program` to cover every program, add `--include-build-dirs` to force a full
rebuild too, and omit `--yes` for a dry run.

Worked example — a new dace transformation, measured on one dycore program:

```bash
python -m gt4py.next.gt_cache_manager status --program 'apply_divergence_damping*'
# -> REPLAY (recompiles, does NOT re-translate)
python -m gt4py.next.gt_cache_manager delete --program 'apply_divergence_damping*' --yes
python -m gt4py.next.gt_cache_manager status --program 'apply_divergence_damping*' --fail-if-cached
# -> RE-TRANSLATE, exit 0
srun ...   # now the pass actually runs
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

- `python -m gt4py.next.gt_cache_manager list|show` inspects entries (program,
  backend, SDFG name, state and map counts).
- `scripts/python/dace_determinism.py` answers the neighbouring question: whether
  dace codegen is deterministic across two runs.
