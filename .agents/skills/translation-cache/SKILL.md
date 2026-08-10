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
gt4py-next-cache status --program '<name glob>' --fail-if-cached
```

The `status` call is the gate: it exits non-zero while any matched program would
still replay. Require it to pass **before** launching an expensive job. Drop
`--program` to cover every program and add `--include-build-dirs` to force a full
rebuild too.

`status` reports the two caches separately, because they hit and miss
independently: `TRANSLATION` is `REPLAY` or `re-translate`, `BUILD` is `reuse` or
`recompile`. Only the translation column decides whether a changed pass runs.

Without `--yes`, `delete` lists what it matched and asks for confirmation on a
terminal; use `--dry-run` to preview and `--yes` in scripts and job files, where
there is no terminal to ask.

`gt4py-next-cache` is installed with gt4py; `python -m gt4py.next.gt_cache_manager`
is the same tool, for when the console script is not on `PATH`.

Worked example — a new dace transformation, measured on one dycore program:

```bash
gt4py-next-cache status --program 'apply_divergence_damping*'
# -> TRANSLATION: REPLAY   BUILD: reuse
gt4py-next-cache delete --program 'apply_divergence_damping*' --yes
gt4py-next-cache status --program 'apply_divergence_damping*' --fail-if-cached
# -> TRANSLATION: re-translate, exit 0
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

- `gt4py-next-cache list|show` inspects entries (program,
  backend, SDFG name, state and map counts).
- `scripts/python/dace_determinism.py` answers the neighbouring question: whether
  dace codegen is deterministic across two runs.
