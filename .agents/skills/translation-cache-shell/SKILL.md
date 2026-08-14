---
name: translation-cache-shell
description: Inspect and prune a gt4py.next cache with shell tools only — find, grep, rm. TRIGGER when clearing or examining a `.gt4py_cache` on a machine where gt4py cannot be imported (a compute node, a container, a cache copied from elsewhere), or when a benchmark must be cleared of cached translations before it runs. SKIP for `gt4py.cartesian` caches, which are a separate mechanism with a different layout.
---

# translation-cache-shell

gt4py.next caches the *translation* step — the optimized SDFG for dace, the
generated source for gtfn — as one pickle per translated program, keyed by the
program and `gt4py.__version__` rather than by the gt4py sources. Editing a
transformation or an optimization pass leaves that key intact, so the next run
replays the cached translation while the build step still recompiles from it.
Clearing only the build folder is not enough, and a fresh library mtime is
evidence of recompilation, never of re-translation.

These recipes need nothing but a shell. They take no lock — see the last section.

## Where the cache is

`.gt4py_cache`, next to the working directory of the run, or under
`GT4PY_BUILD_CACHE_DIR` if that is set. It is only written there when
`GT4PY_BUILD_CACHE_LIFETIME=persistent`; otherwise it lives in a temporary
directory that is deleted when the process exits. See
[`config.py`](../../../src/gt4py/next/config.py) for the defaults.

Pass absolute paths to the commands below. `grep -r` echoes paths exactly as
they were given, so a relative operand yields relative output, which a later
`cd` — or a pipeline run from elsewhere — silently redirects.

Every cached translation is a `*.pkl`, and only translations are — build folders
contain none. So `--include='*.pkl'` scopes to entries alone, whatever directory
layout the gt4py version uses:

```bash
find <cache> -name '*.pkl' | wc -l                          # how many entries
find <cache> -name '*.pkl' -printf '%TF %10s %p\n' | sort   # by date, with sizes
```

## Which entries belong to a program

The program name is stored as a plain string inside each entry:

```bash
grep -rlaF '<program>' <cache> --include='*.pkl'
```

Both flags matter:

- **`-a`** — the entries are binary. Without it `ugrep`, a common drop-in `grep`,
  skips them and prints nothing at all.
- **`-F`** — fixed string, nothing cleverer (see below).

This **over-matches**: a name that contains another name returns both.

### Do not make it word-boundary-aware

`grep -rlaP '(?<!\w)<program>(?!\w)'` looks like the fix for the over-matching
and is silently wrong. Pickle stores a short string as
`0x8c <length-byte> <bytes>`, so the byte before the name **is its length**. When
the name is 48–57, 65–90 or 97–122 characters long that length byte is itself
alphanumeric, the lookbehind rejects the match, and grep reports **nothing** for
that program. Measured on a real 224-entry cache: 3 of 22 programs silently
returned zero hits, the three whose names are 48, 48 and 49 characters. A POSIX
`[^A-Za-z0-9_]` variant returned zero for all 22.

Nothing found reads as nothing cached, which is exactly the wrong way to be
wrong here.

## Clear them

```bash
grep -rlaF '<program>' <cache> --include='*.pkl'                    # look first
grep -rlaF --null '<program>' <cache> --include='*.pkl' | xargs -0 -r rm -v
```

`--null` and `-0` because a cache path containing a space otherwise splits and
`rm` deletes nothing while reporting three failures; `-r` because an empty match
set otherwise runs `rm` with no arguments. Both `ugrep` and GNU grep accept
`--null` (do **not** reach for `-Z`: in `ugrep` that means fuzzy matching).

To clear everything, delete the cache folder.

## Name a single entry

Best effort, for reading a cache you did not write:

```bash
head -c 2000 <entry>.pkl | strings -n 4 | grep -A1 -m1 '^SymbolName$' | tail -1 |
  sed 's/^[^A-Za-z_]//'
```

Exact on all 331 entries of two real caches, whose names run 14–58 characters.
The leading `sed` strips the length byte, which `strings` shows only when it is
printable; a name of 65 characters or more would leave a spurious leading letter.
Treat the output as a label, not as an identity.

## What this cannot do

The runtime holds a lock around each cache file while writing it
(`gt4py._core.locking`). These recipes do not, so an `rm` here can race a
concurrent writer. On a shared filesystem with MPI ranks compiling, clear the
cache between jobs rather than during one.

When the cache is out of reach entirely, invalidate it without touching it by
salting the key:

```bash
export GT4PY_BUILD_CACHE_VERSION_ID=$(git -C <gt4py checkout> rev-parse HEAD)
```

That invalidates the build folders too, and needs no filesystem access at all.
Use a nonce instead of the commit hash when iterating on uncommitted changes.
