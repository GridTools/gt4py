#!/usr/bin/env python3
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""GT4Py / DaCe codegen determinism check.

Library + CLI for verifying that gt4py's DaCe backend produces
byte-identical generated source files across two runs of the same
test selection. Used by the ``test_*_determinism`` nox sessions in
``noxfile.py``; also runnable standalone for ad-hoc comparison of
two existing caches.

The check compares everything dace writes as generated source under
each cached program's ``src/`` — ``cpu/`` and ``cuda/`` (with HIP
picked up automatically under ``cuda/hip/``). It deliberately
ignores SDFGs, build artifacts, source maps, and runtime metadata.

If a snapshot ever encounters a top-level backend other than cpu or
cuda (mpi, sve, mlir, snitch, ...), it errors with a clear message
rather than silently skipping.

As a library
------------

::

    from scripts.dace_deterministic_codegen import check_determinism

    check_determinism(
        cache1=Path(".../run1/.gt4py_cache"),
        cache2=Path(".../run2/.gt4py_cache"),
        diffs_dir=Path(".../diffs"),    # optional
        report_path=Path(".../report.txt"),  # optional
    )

Raises ``DeterminismError`` on mismatch, ``NoProgramsObservedError``
if both caches are empty, ``NoSourceFilesObservedError`` if programs
were cached but contain no source files (typically a missing
``DACE_compiler_build_folder_mode=development``), or
``UnsupportedBackendError`` if the codegen produced an unfamiliar
backend layout.

As a CLI
--------

::

    python scripts/dace_deterministic_codegen.py \\
        --run1 path/to/cache1/.gt4py_cache \\
        --run2 path/to/cache2/.gt4py_cache \\
        [--diffs-dir DIR] [--report FILE]

Exit codes:

    0   codegen is deterministic
    1   codegen differs (see diffs/ and report.txt)
    2   bad arguments / unsupported backend / no source files captured
    3   no programs observed in either cache
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: GT4Py names each cached program folder ``<program_name>_<sha256>``.
PROGRAM_FOLDER_RE = re.compile(r"^(?P<name>.+)_(?P<digest>[0-9a-f]{64})$")

#: The single directory under each program folder we compare. Only ``src/``,
#: nothing else — by design. dace also writes ``include/``, ``sample/``,
#: ``program.sdfg``, source maps under ``map/``, runtime metadata
#: (``dace.conf``, ``*.csv``), and build artifacts under ``build/``. None of
#: those are the codegen surface we care about for this check.
CODEGEN_ROOT = "src"

#: Backends recognized as direct children of ``src/``. dace lays out
#: codegen as ``src/<target_name>/[<target_type>/]<file>``:
#:
#:   - CPU codegen  -> src/cpu/<file>.cpp
#:   - CUDA codegen -> src/cuda/<file>.cu
#:   - HIP  codegen -> src/cuda/hip/<file>.cpp   (NOTE: under cuda/)
#:
#: HIP is dispatched by dace's CUDA target with ``target_type="hip"``, so
#: it lands as a *subdirectory* of ``src/cuda/``, not its own top-level
#: backend folder. That means {cpu, cuda} as a top-level allowlist is
#: enough to cover all three: cpu via ``cpu/``, cuda + hip both via
#: ``cuda/`` (with ``rglob`` picking up the nested hip files).
#:
#: If a snapshot ever encounters another top-level backend (mpi, sve,
#: mlir, snitch, ...), the checker fails loudly rather than silently
#: ignoring — those would need explicit support added here.
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"cpu", "cuda"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnsupportedBackendError(RuntimeError):
    """A program's ``src/`` contained a top-level backend other than cpu/cuda."""


class NoProgramsObservedError(RuntimeError):
    """Neither cache contained any cached program folders."""


class NoSourceFilesObservedError(RuntimeError):
    """Programs were observed in the caches but none contained any source files.

    Almost always means dace's build folder mode was left at ``production``,
    which strips the generated ``src/`` tree after compilation. Set
    ``DACE_compiler_build_folder_mode=development`` (lowercase matters)
    before running the tests so the codegen surface survives into the
    cache and there's actually something to compare.
    """


class DeterminismError(RuntimeError):
    """Two snapshots compared non-identical. ``.results`` carries the details."""

    def __init__(self, message: str, results: list[ProgramResult]) -> None:
        super().__init__(message)
        self.results = results


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


def snapshot_run(cache_root: Path) -> dict[str, ProgramSnapshot]:
    """Walk a gt4py build cache and snapshot every program's generated source.

    The input directory's name is irrelevant — the function looks for
    immediate subdirectories matching ``<name>_<64-char-hex-digest>``
    (gt4py's program-folder naming) and reads ``<program>/src/``
    recursively under each one. HIP files at ``src/cuda/hip/`` are
    picked up automatically by the recursive walk.

    Returns an empty dict (rather than raising) when the path does not
    exist, is not a directory, has no subdirectories, or has only
    subdirectories whose names don't match the program-folder pattern.
    Callers can use :func:`_diagnose_empty_cache` to get a human
    description of which of those it was.

    Raises :class:`UnsupportedBackendError` if any program's ``src/``
    contains a top-level backend not in :data:`SUPPORTED_BACKENDS`.
    """
    if not cache_root.is_dir():
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
            # downstream will flag it if its counterpart has files.
            out[name] = ProgramSnapshot(name=name, folder=folder, files={})
            continue

        # Backend check: every direct child of src/ must be a supported
        # top-level backend. HIP lives nested under cuda/, so cuda is
        # what matters here, not "hip".
        for bd in sorted(d for d in src_root.iterdir() if d.is_dir()):
            if bd.name not in SUPPORTED_BACKENDS:
                raise UnsupportedBackendError(
                    f"unsupported dace backend `{bd.name}/` found under "
                    f"{src_root} — this checker currently supports "
                    f"{sorted(SUPPORTED_BACKENDS)} as top-level backends "
                    f"(HIP is handled under `cuda/hip/`). Add explicit "
                    f"support in scripts/dace_deterministic_codegen.py "
                    f"before running this selection."
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


def _diagnose_empty_cache(cache_root: Path) -> str:
    """Return a one-line explanation of why :func:`snapshot_run` found nothing.

    Distinguishes "path didn't exist", "path is a file", "directory has
    no subdirectories", and "directory has subdirectories but none match
    the program-folder pattern" (with a sample of the names found, so a
    wrong-path mistake — e.g. passing the parent of ``.gt4py_cache`` —
    is immediately visible).
    """
    if not cache_root.exists():
        return "path does not exist"
    if not cache_root.is_dir():
        return "path exists but is not a directory"
    subdirs = sorted(p for p in cache_root.iterdir() if p.is_dir())
    if not subdirs:
        return "directory has no subdirectories"
    matching = [p for p in subdirs if PROGRAM_FOLDER_RE.match(p.name)]
    if not matching:
        sample_names = [p.name for p in subdirs[:3]]
        suffix = f" (and {len(subdirs) - 3} more)" if len(subdirs) > 3 else ""
        return (
            f"directory contains {len(subdirs)} subdirectory(ies) but none "
            f"match the program-folder pattern `<name>_<64-char-hex-digest>/` "
            f"(saw: {sample_names}{suffix}). If one of those names is "
            f"`.gt4py_cache`, you've passed the parent of the cache by mistake."
        )
    return "directory has matching subdirectories but they were filtered out"


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
    """Pair programs by name across the two snapshots and diff their files."""
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
    """Emit a unified diff per differing file under ``diffs_dir/<program>/``."""
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
            lines.extend(f"           differs: {rel}" for rel in r.differing_files)
            lines.extend(f"           only in run1: {rel}" for rel in r.only_in_run1)
            lines.extend(f"           only in run2: {rel}" for rel in r.only_in_run2)

    lines.append("")
    if n_total == 0:
        lines.append("RESULT: no programs observed (nothing was cached).")
    elif n_diff == 0:
        lines.append(f"RESULT: codegen deterministic — {n_match} program(s) match.")
    else:
        lines.append(f"RESULT: NON-DETERMINISTIC CODEGEN — {n_diff}/{n_total} program(s) differ.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Library entry point
# ---------------------------------------------------------------------------


def check_determinism(
    cache1: Path,
    cache2: Path,
    *,
    diffs_dir: Path | None = None,
    report_path: Path | None = None,
) -> list[ProgramResult]:
    """Compare two gt4py caches; write artifacts; raise on mismatch.

    Snapshots both caches (under ``<program>/src/`` recursively) and
    diffs them. Optionally writes per-file unified diffs to
    ``diffs_dir/<program>/`` and a human-readable summary to
    ``report_path``.

    Returns the list of :class:`ProgramResult` on a successful match.

    Raises:
        UnsupportedBackendError:
            A snapshot contained a backend other than cpu/cuda.
        NoProgramsObservedError:
            Both caches were empty — likely zero tests collected.
        DeterminismError:
            One or more programs differed between the two runs.
    """
    snap1 = snapshot_run(cache1)
    snap2 = snapshot_run(cache2)
    results = compare(snap1, snap2)

    if diffs_dir is not None:
        write_diffs(results, snap1, snap2, diffs_dir)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(results))

    if not results:
        diag1 = _diagnose_empty_cache(cache1)
        diag2 = _diagnose_empty_cache(cache2)
        raise NoProgramsObservedError(
            "no programs observed in either cache:\n"
            f"  run1 ({cache1}): {diag1}\n"
            f"  run2 ({cache2}): {diag2}"
        )

    # Safety net for the silent-false-positive case where both runs cached
    # programs but every program's src/ tree is empty — typically because
    # dace's build_folder_mode is `production` (the gt4py default). Without
    # this, the comparator would see {} == {} for every program and report
    # `deterministic` despite there being nothing to compare.
    total_files = sum(len(s.files) for s in snap1.values()) + sum(
        len(s.files) for s in snap2.values()
    )
    if total_files == 0:
        raise NoSourceFilesObservedError(
            f"{len(results)} program(s) cached, but none of them contain any "
            f"source files under src/. This almost always means dace's build "
            f"folder mode is `production` rather than `development`, which "
            f"strips the codegen output after compilation. Set "
            f"DACE_compiler_build_folder_mode=development (lowercase matters) "
            f"before running the tests so src/cpu/*.cpp and src/cuda/*.cu "
            f"survive into the cache."
        )

    n_diff = sum(1 for r in results if not r.match)
    if n_diff > 0:
        raise DeterminismError(
            f"DaCe codegen is non-deterministic: {n_diff}/{len(results)} program(s) differ",
            results,
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dace_deterministic_codegen",
        description=(
            "Compare two gt4py build caches and check whether the DaCe "
            "generated source files are byte-identical between them."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run1",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the first .gt4py_cache directory.",
    )
    p.add_argument(
        "--run2",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the second .gt4py_cache directory.",
    )
    p.add_argument(
        "--diffs-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, write per-file unified diffs to this directory.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, write the human-readable summary report to this file.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        results = check_determinism(
            args.run1.expanduser().resolve(),
            args.run2.expanduser().resolve(),
            diffs_dir=args.diffs_dir.expanduser().resolve() if args.diffs_dir else None,
            report_path=args.report.expanduser().resolve() if args.report else None,
        )
    except UnsupportedBackendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except NoProgramsObservedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except NoSourceFilesObservedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except DeterminismError as e:
        print(render_report(e.results))
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(render_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
