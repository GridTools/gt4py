#!/usr/bin/env python3
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""GT4Py / DaCe codegen determinism check.

Drives an icon4py test selection through nox **twice** with isolated
gt4py build caches, then checks that the generated source files under
each program's `src/` are byte-identical between the two runs. A diff
is a determinism bug.

Compares only the contents of `<program>/src/` — the actual generated
backend code. Currently supports cpu, cuda, and hip (hip is emitted by
dace under `src/cuda/hip/`). Any other top-level backend under `src/`
(mpi, sve, mlir, snitch, …) causes the harness to fail with a clear
message rather than silently ignore it.

Valid `--selection` and `--component` values are read from icon4py's
own `noxfile.py` at runtime (no hardcoding here), so the harness
tracks any future changes to icon4py's parametrization automatically.

Mirrors icon4py's `ci/dace.yml` invocation pattern, with the session
name configurable:

    nox -r -s "<session>-<py>(<selection>, <component>)" -- <pass-through args>

Defaults to `<session>=test_model`, which is the icon4py main test
entry point and what `ci/dace.yml` uses.

Outputs land at `<icon4py>/_dace_deterministic_codegen/`:
    run1/.gt4py_cache/...       run1/test.log
    run2/.gt4py_cache/...       run2/test.log
    diffs/<program>/<file>.diff   (only on mismatch)
    report.txt
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: GT4Py names each cached program folder `<program_name>_<sha256>`.
PROGRAM_FOLDER_RE = re.compile(r"^(?P<name>.+)_(?P<digest>[0-9a-f]{64})$")

#: The single directory under each program folder we compare. Only `src/`,
#: nothing else — by design. dace also writes `include/`, `sample/`,
#: `program.sdfg`, source maps under `map/`, runtime metadata
#: (`dace.conf`, `*.csv`), and build artifacts under `build/`. None of
#: those are the codegen surface we care about for this check.
CODEGEN_ROOT = "src"

#: Backends recognized as direct children of `src/`. dace lays out
#: codegen as `src/<target_name>/[<target_type>/]<file>`:
#:
#:   - CPU codegen  → src/cpu/<file>.cpp
#:   - CUDA codegen → src/cuda/<file>.cu
#:   - HIP  codegen → src/cuda/hip/<file>.cpp   (NOTE: under cuda/)
#:
#: HIP is dispatched by dace's CUDA target with `target_type="hip"`, so
#: it lands as a *subdirectory* of `src/cuda/`, not its own top-level
#: backend folder. That means {cpu, cuda} as a top-level allowlist is
#: enough to cover all three: cpu via `cpu/`, cuda + hip both via
#: `cuda/` (with `rglob` picking up the nested hip files).
#:
#: If a snapshot ever encounters another top-level backend (mpi, sve,
#: mlir, snitch, …), the harness fails loudly rather than silently
#: ignoring — those would need explicit support added here.
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"cpu", "cuda"})

#: Where outputs are written, relative to the icon4py checkout.
WORKDIR_NAME = "_dace_deterministic_codegen"


# ---------------------------------------------------------------------------
# icon4py noxfile introspection
# ---------------------------------------------------------------------------


class NoxfileIntrospectionError(RuntimeError):
    """Raised when we can't extract sessions/components from the noxfile."""


def introspect_icon4py_noxfile(
    noxfile: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    """Parse icon4py's noxfile.py and extract the valid `selection` and
    `component` values. Returns `(selections, components)`.

    Reads the noxfile as AST — does not execute it. Two reasons:
    importing would require `nox` in this script's environment, and
    noxfile imports often have side effects (icon4py's pulls in a
    handful of typing imports plus nox's own session machinery).

    Looks for two type-alias definitions matching icon4py main:

        ModelTestsSubset:    TypeAlias = Literal["datatest", "stencils", "basic"]
        ModelSubpackagePath: TypeAlias = Literal["atmosphere/advection", ...]

    Components are derived from the *leaf name* of each subpackage path
    (`subpackage.split("/")[-1]`), matching the `id=...` icon4py uses
    in nox.param. So `atmosphere/subgrid_scale_physics/muphys` becomes
    the component `muphys`.
    """
    if not noxfile.is_file():
        raise NoxfileIntrospectionError(
            f"no noxfile.py at {noxfile} — is --icon4py the icon4py repo root?"
        )

    try:
        tree = ast.parse(noxfile.read_text())
    except SyntaxError as e:
        raise NoxfileIntrospectionError(f"could not parse {noxfile} as Python: {e}") from e

    selections = _extract_literal_strings(tree, "ModelTestsSubset")
    subpackages = _extract_literal_strings(tree, "ModelSubpackagePath")

    if not selections:
        raise NoxfileIntrospectionError(
            f"could not find `ModelTestsSubset: TypeAlias = Literal[...]` "
            f"in {noxfile}. icon4py's noxfile structure may have changed."
        )
    if not subpackages:
        raise NoxfileIntrospectionError(
            f"could not find `ModelSubpackagePath: TypeAlias = Literal[...]` "
            f"in {noxfile}. icon4py's noxfile structure may have changed."
        )

    components = frozenset(p.rsplit("/", 1)[-1] for p in subpackages)
    return frozenset(selections), components


def _extract_literal_strings(tree: ast.AST, alias_name: str) -> list[str]:
    """Find `<alias_name>: TypeAlias = Literal["a", "b", ...]` in the AST
    and return the string literals. Returns [] if not found or shape is
    unexpected (caller decides whether that's fatal)."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        if node.target.id != alias_name:
            continue
        # Match the AST pattern for Literal["a", "b", ...]:
        # a Subscript whose value is the Name "Literal" and whose slice is
        # either a Tuple of string Constants or a single string Constant.
        v = node.value
        if not isinstance(v, ast.Subscript):
            continue
        elts: list[ast.expr] = []
        if isinstance(v.slice, ast.Tuple):
            elts = list(v.slice.elts)
        else:
            # Single-arg Literal["foo"]
            elts = [v.slice]
        out: list[str] = []
        for e in elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                out.append(e.value)
        return out
    return []


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FileEntry:
    relpath: str
    sha256: str


@dataclasses.dataclass
class ProgramSnapshot:
    name: str
    folder: Path
    files: dict[str, FileEntry]


class UnsupportedBackendError(RuntimeError):
    """A program's `src/` contained a top-level backend other than cpu/cuda."""


def snapshot_run(cache_root: Path) -> dict[str, ProgramSnapshot]:
    """Walk a `.gt4py_cache` and snapshot every program's generated source.

    For each `<program>_<digest>/` folder, we read everything under
    `<program>/src/` recursively. dace lays this out as
    `src/<target_name>/[<target_type>/]<file>`:

        src/cpu/<file>.cpp
        src/cuda/<file>.cu        (CUDA — target_type="")
        src/cuda/hip/<file>.cpp   (HIP  — target_type="hip", under cuda/)

    Currently supports cpu and cuda as top-level backends. HIP is
    handled implicitly because dace nests it inside `src/cuda/hip/`,
    not as a separate top-level directory; the recursive walk picks
    it up automatically.

    If we encounter any *other* top-level backend under `src/` (mpi,
    sve, mlir, snitch, ...), raises UnsupportedBackendError so the
    user knows immediately rather than silently skipping.
    """
    if not cache_root.exists():
        return {}

    out: dict[str, ProgramSnapshot] = {}
    for folder in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        m = PROGRAM_FOLDER_RE.match(folder.name)
        if not m:
            continue
        name = m.group("name")

        src_root = folder / CODEGEN_ROOT
        if not src_root.is_dir():
            # No src/ at all — record an empty snapshot. Pairing logic
            # downstream will flag it if its counterpart in the other run
            # has files.
            out[name] = ProgramSnapshot(name=name, folder=folder, files={})
            continue

        # Backend check: every direct child of src/ must be a supported
        # top-level backend. HIP lives nested under cuda/, so cuda is
        # what matters here, not "hip".
        backend_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
        for bd in backend_dirs:
            if bd.name not in SUPPORTED_BACKENDS:
                raise UnsupportedBackendError(
                    f"unsupported dace backend `{bd.name}/` found under "
                    f"{src_root} — this harness currently supports "
                    f"{sorted(SUPPORTED_BACKENDS)} as top-level backends "
                    f"(HIP is handled under `cuda/hip/`). Add explicit "
                    f"support in dace_deterministic_codegen.py before "
                    f"running this selection."
                )

        # rglob recursively descends — picks up `cuda/hip/<file>` along
        # with `cpu/<file>` and `cuda/<file>`, no special-casing needed.
        files: dict[str, FileEntry] = {}
        for fpath in sorted(src_root.rglob("*")):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(folder).as_posix()
            files[rel] = FileEntry(relpath=rel, sha256=_sha256(fpath))
        out[name] = ProgramSnapshot(name=name, folder=folder, files=files)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProgramResult:
    name: str
    match: bool
    differing_files: list[str]
    only_in_run1: list[str]
    only_in_run2: list[str]


def compare(
    snap1: dict[str, ProgramSnapshot],
    snap2: dict[str, ProgramSnapshot],
) -> list[ProgramResult]:
    results: list[ProgramResult] = []
    for name in sorted(set(snap1) | set(snap2)):
        s1 = snap1.get(name)
        s2 = snap2.get(name)

        if s1 is None or s2 is None:
            results.append(
                ProgramResult(
                    name=name,
                    match=False,
                    differing_files=[],
                    only_in_run1=sorted((s1.files if s1 else {}).keys()),
                    only_in_run2=sorted((s2.files if s2 else {}).keys()),
                )
            )
            continue

        keys1, keys2 = set(s1.files), set(s2.files)
        only1 = sorted(keys1 - keys2)
        only2 = sorted(keys2 - keys1)
        differing = sorted(
            rel for rel in keys1 & keys2 if s1.files[rel].sha256 != s2.files[rel].sha256
        )
        results.append(
            ProgramResult(
                name=name,
                match=not (differing or only1 or only2),
                differing_files=differing,
                only_in_run1=only1,
                only_in_run2=only2,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Diff + report
# ---------------------------------------------------------------------------


def write_diffs(
    results: list[ProgramResult],
    snap1: dict[str, ProgramSnapshot],
    snap2: dict[str, ProgramSnapshot],
    diffs_dir: Path,
) -> None:
    for r in results:
        if r.match:
            continue
        s1, s2 = snap1.get(r.name), snap2.get(r.name)
        prog_dir = diffs_dir / r.name
        for rel in r.differing_files:
            f1 = (s1.folder / rel) if s1 else None
            f2 = (s2.folder / rel) if s2 else None
            if not (f1 and f2 and f1.exists() and f2.exists()):
                continue
            try:
                t1 = f1.read_text().splitlines(keepends=True)
                t2 = f2.read_text().splitlines(keepends=True)
            except UnicodeDecodeError:
                prog_dir.mkdir(parents=True, exist_ok=True)
                (prog_dir / f"{rel.replace('/', '__')}.binary-differs").write_text(
                    f"binary content differs:\n  run1: {f1}\n  run2: {f2}\n"
                )
                continue
            udiff = "".join(
                difflib.unified_diff(
                    t1,
                    t2,
                    fromfile=f"run1/{rel}",
                    tofile=f"run2/{rel}",
                    n=3,
                )
            )
            prog_dir.mkdir(parents=True, exist_ok=True)
            (prog_dir / f"{rel.replace('/', '__')}.diff").write_text(udiff)


def render_report(results: list[ProgramResult]) -> str:
    n_total = len(results)
    n_match = sum(1 for r in results if r.match)
    n_diff = n_total - n_match

    lines = [f"Programs: {n_total}    matches: {n_match}    mismatches: {n_diff}", ""]
    for r in results:
        lines.append(f"  [{'MATCH ' if r.match else 'DIFFER'}] {r.name}")
        if not r.match:
            for rel in r.differing_files:
                lines.append(f"           differs: {rel}")
            for rel in r.only_in_run1:
                lines.append(f"           only in run1: {rel}")
            for rel in r.only_in_run2:
                lines.append(f"           only in run2: {rel}")

    lines.append("")
    if n_total == 0:
        lines.append("RESULT: no programs observed (nothing was cached).")
    elif n_diff == 0:
        lines.append(f"RESULT: codegen deterministic — {n_match} program(s) match.")
    else:
        lines.append(f"RESULT: NON-DETERMINISTIC CODEGEN — {n_diff}/{n_total} program(s) differ.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Nox runner
# ---------------------------------------------------------------------------


def run_nox(
    icon4py: Path,
    run_dir: Path,
    log_path: Path,
    session: str,
    selection: str,
    component: str,
    python: str,
    posargs: list[str],
) -> int:
    """Run nox once with `GT4PY_BUILD_CACHE_DIR=run_dir`. Returns the exit code.

    Mirrors `ci/dace.yml`: positional session ID, `-r` to reuse the venv
    between runs (so run1 and run2 see identical venv state — important
    for the determinism check).

    NOTE: gt4py's config appends `.gt4py_cache` to GT4PY_BUILD_CACHE_DIR,
    so `run_dir` is the *parent*: gt4py creates
    `run_dir/.gt4py_cache/<program>_<digest>/` inside it.
    """
    session_id = f"{session}-{python}({selection}, {component})"
    argv = ["nox", "-r", "-s", session_id]
    if posargs:
        argv.append("--")
        argv.extend(posargs)

    env = dict(os.environ.items())
    env["GT4PY_BUILD_CACHE_DIR"] = str(run_dir)
    env["GT4PY_BUILD_CACHE_LIFETIME"] = "persistent"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        logf.write(
            f"# cwd:    {icon4py}\n"
            "# command:\n  " + "\n  ".join(repr(a) for a in argv) + "\n"
            f"# GT4PY_BUILD_CACHE_DIR={run_dir}\n"
            f"#   (gt4py appends .gt4py_cache; cache lands at {run_dir}/.gt4py_cache/)\n"
            "# ---\n"
        )
        logf.flush()
        proc = subprocess.run(
            argv, cwd=str(icon4py), env=env, stdout=logf, stderr=subprocess.STDOUT
        )
    return proc.returncode


# ---------------------------------------------------------------------------
# Workdir
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Workdir:
    """Two parent dirs for gt4py's cache + a place for logs/diffs/report."""

    root: Path

    @property
    def run1_dir(self) -> Path:
        return self.root / "run1"

    @property
    def run2_dir(self) -> Path:
        return self.root / "run2"

    @property
    def cache1(self) -> Path:
        return self.run1_dir / ".gt4py_cache"

    @property
    def cache2(self) -> Path:
        return self.run2_dir / ".gt4py_cache"

    @property
    def log1(self) -> Path:
        return self.run1_dir / "test.log"

    @property
    def log2(self) -> Path:
        return self.run2_dir / "test.log"

    @property
    def diffs(self) -> Path:
        return self.root / "diffs"

    @property
    def report(self) -> Path:
        return self.root / "report.txt"

    def prepare(self) -> None:
        """Wipe stale state from previous invocations."""
        for d in (self.run1_dir, self.run2_dir, self.diffs):
            if d.exists():
                shutil.rmtree(d)
        for d in (self.run1_dir, self.run2_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.report.exists():
            self.report.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dace_deterministic_codegen",
        description=(
            "Run an icon4py test selection twice via nox with isolated gt4py "
            "caches and check that the generated source code is byte-identical."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--icon4py",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Path to icon4py checkout. Accepts BOTH absolute and relative "
            "paths. Relative paths are resolved against the current working "
            "directory (i.e. wherever you invoke this script from)."
        ),
    )
    p.add_argument(
        "--session",
        default="test_model",
        metavar="NAME",
        help=(
            "Nox session name. Composed with --python/--selection/--component "
            "into the final session ID `<session>-<python>(<selection>, "
            "<component>)`. Default matches icon4py's main test entry point."
        ),
    )
    p.add_argument(
        "--selection",
        required=True,
        metavar="NAME",
        help=(
            "icon4py noxfile selection (e.g. stencils, datatest, basic). "
            "Validated at runtime against icon4py's actual noxfile."
        ),
    )
    p.add_argument(
        "--component",
        required=True,
        metavar="NAME",
        help=(
            "icon4py noxfile subpackage leaf name (e.g. muphys, dycore). "
            "Validated at runtime against icon4py's actual noxfile."
        ),
    )
    p.add_argument(
        "--python",
        default="3.10",
        metavar="X.Y",
        help="Python version for the nox session.",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Where run1/, run2/, diffs/, and report.txt are written. "
            "Accepts absolute or relative paths (resolved against cwd). "
            "If the directory already exists from a prior run, its contents "
            "are wiped before this run starts — no merging or appending. "
            "Default: <icon4py>/_dace_deterministic_codegen/"
        ),
    )
    p.add_argument(
        "--posarg",
        action="append",
        default=[],
        dest="posargs",
        metavar="ARG",
        help=(
            "Forwarded to pytest via `nox -- ARG`. Repeatable. "
            "Example: --posarg=--backend=dace_cpu --posarg=--grid=icon_regional"
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Resolve every path to absolute up-front, so the harness can be run
    # from any cwd. We print what the path resolved to — `--icon4py ../foo`
    # behaves intuitively but it's nice to confirm what it landed on.
    icon4py = args.icon4py.expanduser().resolve()
    if not args.icon4py.is_absolute():
        print(f"--icon4py resolved to: {icon4py}")
    if not icon4py.is_dir():
        print(f"error: --icon4py path is not a directory: {icon4py}", file=sys.stderr)
        return 2
    noxfile_path = icon4py / "noxfile.py"
    if not noxfile_path.is_file():
        print(
            f"error: no noxfile.py at {noxfile_path} — is --icon4py the icon4py repo root?",
            file=sys.stderr,
        )
        return 2

    # Introspect icon4py's noxfile to discover the legal selection /
    # component values. This avoids hardcoding the lists, so the harness
    # auto-tracks any future changes to icon4py's noxfile structure.
    try:
        valid_selections, valid_components = introspect_icon4py_noxfile(noxfile_path)
    except NoxfileIntrospectionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.selection not in valid_selections:
        print(
            f"error: --selection {args.selection!r} is not one of "
            f"{sorted(valid_selections)} (extracted from {noxfile_path})",
            file=sys.stderr,
        )
        return 2
    if args.component not in valid_components:
        print(
            f"error: --component {args.component!r} is not one of "
            f"{sorted(valid_components)} (extracted from {noxfile_path})",
            file=sys.stderr,
        )
        return 2

    workdir_root = (
        args.workdir.expanduser().resolve() if args.workdir is not None else icon4py / WORKDIR_NAME
    )
    if args.workdir is not None and not args.workdir.is_absolute():
        print(f"--workdir resolved to: {workdir_root}")
    workdir = Workdir(root=workdir_root)
    workdir.prepare()  # wipes run1/, run2/, diffs/, report.txt — see prepare()

    session_id = f"{args.session}-{args.python}({args.selection}, {args.component})"

    # ----- Run 1
    print(f"[1/2] nox -s '{session_id}' (cache: {workdir.run1_dir})", flush=True)
    rc1 = run_nox(
        icon4py,
        workdir.run1_dir,
        workdir.log1,
        args.session,
        args.selection,
        args.component,
        args.python,
        args.posargs,
    )
    if rc1 != 0:
        print(f"error: run 1 failed (exit {rc1}). See log: {workdir.log1}", file=sys.stderr)
        return 4

    # ----- Run 2
    print(f"[2/2] nox -s '{session_id}' (cache: {workdir.run2_dir})", flush=True)
    rc2 = run_nox(
        icon4py,
        workdir.run2_dir,
        workdir.log2,
        args.session,
        args.selection,
        args.component,
        args.python,
        args.posargs,
    )
    if rc2 != 0:
        print(f"error: run 2 failed (exit {rc2}). See log: {workdir.log2}", file=sys.stderr)
        return 4

    # ----- Snapshot + compare + report
    try:
        snap1 = snapshot_run(workdir.cache1)
        snap2 = snapshot_run(workdir.cache2)
    except UnsupportedBackendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    results = compare(snap1, snap2)
    write_diffs(results, snap1, snap2, workdir.diffs)
    report = render_report(results)
    workdir.report.write_text(report)
    print(report)
    print(f"workdir: {workdir.root}")

    if not results:
        print(
            f"error: no programs observed in either run — check the logs:\n"
            f"  {workdir.log1}\n  {workdir.log2}",
            file=sys.stderr,
        )
        return 3

    return 1 if any(not r.match for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
