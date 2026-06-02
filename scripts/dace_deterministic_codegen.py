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
test selection. Used by the ``test_next_dace_determinism`` nox session
in ``noxfile.py``; also runnable standalone for ad-hoc comparison of
two existing caches.

The check operates on the ``gt4py.next`` build cache: a flat
``<cache_root>/<program_folder>/src/...`` structure written when
``GT4PY_BUILD_CACHE_DIR`` points at ``<cache_root>``'s parent. Each
immediate sub-directory of ``<cache_root>`` is treated as one cached
program and keyed by its directory name verbatim; the generated source
under that program's ``src/`` sub-tree is hashed and compared between
the two runs. Unknown top-level backends (anything other than cpu/cuda,
with HIP nesting under cuda/hip) raise :class:`UnsupportedBackendError`.

As a library
------------

::

    from scripts.dace_deterministic_codegen import check_determinism

    check_determinism(
        cache1=Path(".../run1/.gt4py_cache"),
        cache2=Path(".../run2/.gt4py_cache"),
        diffs_dir=Path(".../diffs"),         # optional
        report_path=Path(".../report.txt"),  # optional
    )

Raises ``DeterminismError`` on mismatch, ``NoProgramsObservedError``
if both caches are empty, ``NoSourceFilesObservedError`` if programs
were cached but contain no source files (typically a missing
``DACE_compiler_build_folder_mode=development``), or
``UnsupportedBackendError`` if the codegen produced an unfamiliar
top-level backend.

As a CLI
--------

::

    python scripts/dace_deterministic_codegen.py \\
        --run1 path/to/cache1 \\
        --run2 path/to/cache2 \\
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
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Coupling to gt4py.next's cache layout
# ---------------------------------------------------------------------------
#
# This script reads files written by gt4py.next's build cache, so it has to
# know *one* thing about that layout: the name of the sub-directory under a
# program folder that holds the DaCe-generated sources. That single
# assumption is captured here, in one place, on purpose — if gt4py.next ever
# renames it, this is the only line to update.
#
# Everything else is layout-agnostic: programs are discovered as the
# immediate sub-directories of the cache root and keyed by their directory
# name verbatim (no parsing of the name), so the script does not break if the
# program-folder naming scheme changes.

#: Sub-directory under each program folder holding the codegen we compare.
#: Only this directory is read — DaCe also writes ``include/``, ``sample/``,
#: ``program.sdfg``, source maps, runtime metadata (``dace.conf``, ``*.csv``),
#: and build artifacts under ``build/``; none of those are the codegen
#: surface this check cares about.
CODEGEN_SUBDIR = "src"

#: Backends recognized as direct children of ``src/``. DaCe lays out codegen
#: as ``src/<target_name>/[<target_type>/]<file>``:
#:
#:   - CPU codegen  -> src/cpu/<file>.cpp
#:   - CUDA codegen -> src/cuda/<file>.cu
#:   - HIP  codegen -> src/cuda/hip/<file>.cpp   (NOTE: under cuda/)
#:
#: If a snapshot ever encounters another top-level backend,
#: the checker fails loudly rather than silently ignoring it —
#: those would need explicit support added here.
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

    Almost always means DaCe's build folder mode was left at ``production``,
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
    #: Path of the generated source file relative to its program folder; used
    #: as the comparison key across the two runs.
    relpath: str
    #: sha256 of the file's bytes.
    sha256: str
    #: Absolute path to the file on disk, so :func:`write_diffs` can read it
    #: back without re-deriving the location from the program folder.
    abspath: Path


@dataclasses.dataclass
class ProgramSnapshot:
    #: The program folder's directory name (verbatim), used as the program key.
    name: str
    #: Absolute path to the program folder on disk.
    folder: Path
    #: Generated source files, keyed by path relative to ``folder``.
    files: dict[str, FileEntry]


def snapshot_run(cache_root: Path) -> dict[str, ProgramSnapshot]:
    """Walk a gt4py.next build cache and snapshot every program's generated source.

    Every immediate sub-directory of ``cache_root`` is treated as one cached
    program, keyed by its directory name verbatim, and its ``src/`` sub-tree
    is hashed recursively (HIP files at ``src/cuda/hip/`` are picked up by the
    recursive walk). Keying by the full folder name keeps programs that share
    an entry-point name but differ by cache digest distinct, so none silently
    overwrites another.

    Returns an empty dict (rather than raising) when the path doesn't exist or
    contains no programs;

    Raises :class:`UnsupportedBackendError` if any program's ``src/`` contains
    a top-level backend not in :data:`SUPPORTED_BACKENDS`.
    """
    if not cache_root.is_dir():
        return {}

    out: dict[str, ProgramSnapshot] = {}
    for folder in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        name = folder.name

        src_root = folder / CODEGEN_SUBDIR
        if not src_root.is_dir():
            # No src/ at all — record an empty snapshot. Pairing logic
            # downstream will flag it if its counterpart has files.
            out[name] = ProgramSnapshot(name=name, folder=folder, files={})
            continue

        # Backend check: every direct child of src/ must be a supported
        # top-level backend. HIP lives nested under cuda/, so cuda is what
        # matters here, not "hip".
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

        # rglob recursively descends — picks up `cuda/hip/<file>` along with
        # `cpu/<file>` and `cuda/<file>`, no special-casing needed.
        files: dict[str, FileEntry] = {}
        for fpath in sorted(src_root.rglob("*")):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(folder).as_posix()
            files[rel] = FileEntry(relpath=rel, sha256=_sha256(fpath), abspath=fpath)
        out[name] = ProgramSnapshot(name=name, folder=folder, files=files)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _diagnose_empty_cache(cache_root: Path) -> str:
    """Return a one-line explanation of why :func:`snapshot_run` found nothing."""
    if not cache_root.exists():
        return "path does not exist"
    if not cache_root.is_dir():
        return "path exists but is not a directory"
    subdirs = sorted(p for p in cache_root.iterdir() if p.is_dir())
    if not subdirs:
        return "directory has no subdirectories (no programs cached)"
    return "directory has subdirectories but none could be snapshotted"


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProgramResult:
    """Outcome of comparing one program across the two runs.

    ``snapshot1``/``snapshot2`` are the per-run snapshots (``None`` when the
    program was not cached in that run). All status — whether the program is
    present in each run, whether it matches, whether it is missing on one
    side — is derived from these and the file-level diff lists below, so the
    fields cannot drift out of sync.
    """

    name: str
    snapshot1: ProgramSnapshot | None
    snapshot2: ProgramSnapshot | None
    differing_files: list[str]
    only_in_run1: list[str]
    only_in_run2: list[str]

    @property
    def present_in_run1(self) -> bool:
        return self.snapshot1 is not None

    @property
    def present_in_run2(self) -> bool:
        return self.snapshot2 is not None

    @property
    def match(self) -> bool:
        """True iff the program is in both runs with identical source files."""
        return (
            self.present_in_run1
            and self.present_in_run2
            and not self.differing_files
            and not self.only_in_run1
            and not self.only_in_run2
        )

    @property
    def missing_on_one_side(self) -> bool:
        """True iff the program folder was cached in exactly one of the two runs."""
        return self.present_in_run1 ^ self.present_in_run2


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
            # Program cached in only one run: surface the present side's file
            # list under the corresponding only_in_run* so the report shows
            # what that run produced.
            results.append(
                ProgramResult(
                    name=name,
                    snapshot1=s1,
                    snapshot2=s2,
                    differing_files=[],
                    only_in_run1=sorted(s1.files) if s1 else [],
                    only_in_run2=sorted(s2.files) if s2 else [],
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
                snapshot1=s1,
                snapshot2=s2,
                differing_files=differing,
                only_in_run1=only1,
                only_in_run2=only2,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Diff + report
# ---------------------------------------------------------------------------


def write_diffs(results: list[ProgramResult], diffs_dir: Path) -> None:
    """Emit a unified diff per differing file under ``<diffs_dir>/<program>/``."""
    for r in results:
        if not r.differing_files:
            continue
        # A program can only have differing files when it is present in both
        # runs, so both snapshots are guaranteed non-None here.
        assert r.snapshot1 is not None and r.snapshot2 is not None
        prog_dir = diffs_dir / r.name
        for rel in r.differing_files:
            f1 = r.snapshot1.files[rel].abspath
            f2 = r.snapshot2.files[rel].abspath
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
                difflib.unified_diff(t1, t2, fromfile=f"run1/{rel}", tofile=f"run2/{rel}", n=3)
            )
            prog_dir.mkdir(parents=True, exist_ok=True)
            (prog_dir / f"{rel.replace('/', '__')}.diff").write_text(udiff)


def render_report(results: list[ProgramResult], *, tolerate_missing: bool = False) -> str:
    n_total = len(results)
    n_missing = sum(1 for r in results if r.missing_on_one_side)
    n_diff_content = sum(
        1 for r in results if r.differing_files or (not r.match and not r.missing_on_one_side)
    )
    n_match = n_total - n_missing - n_diff_content

    header = (
        f"Programs: {n_total}    matches: {n_match}    "
        f"differs: {n_diff_content}    only-in-one-run: {n_missing}"
    )
    lines = [header, ""]
    for r in results:
        if r.match:
            tag = "MATCH "
        elif r.missing_on_one_side:
            tag = "ONE-OF"
        else:
            tag = "DIFFER"
        lines.append(f"  [{tag}] {r.name}")
        if not r.match:
            lines.extend(f"           differs: {rel}" for rel in r.differing_files)
            lines.extend(f"           only in run1: {rel}" for rel in r.only_in_run1)
            lines.extend(f"           only in run2: {rel}" for rel in r.only_in_run2)

    lines.append("")
    if n_total == 0:
        lines.append("RESULT: no programs observed (nothing was cached).")
    elif n_diff_content == 0 and n_missing == 0:
        lines.append(f"RESULT: codegen deterministic — {n_match} program(s) match.")
    elif n_diff_content == 0 and tolerate_missing:
        lines.append(
            f"RESULT: codegen deterministic across the {n_match} shared program(s); "
            f"{n_missing} program(s) cached in only one run (tolerated)."
        )
    else:
        suffix = f" (plus {n_missing} cached in only one run)" if n_missing else ""
        lines.append(
            f"RESULT: NON-DETERMINISTIC CODEGEN — {n_diff_content}/{n_total} "
            f"program(s) differ by content{suffix}."
        )
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
    tolerate_missing: bool = True,
) -> list[ProgramResult]:
    """Compare two gt4py.next caches; write artifacts; raise on mismatch.

    Snapshots both caches and diffs them. Optionally writes per-file unified
    diffs to ``<diffs_dir>/<program>/`` and a human-readable summary to
    ``report_path``.

    Returns the list of :class:`ProgramResult` on a successful match.

    Parameters
    ----------
    cache1, cache2
        The two ``.gt4py_cache/`` directories to compare (each the parent of
        all the program folders).
    tolerate_missing
        See module docstring. Default ``True`` (lenient).
    diffs_dir, report_path
        If set, persist diagnostic artifacts.

    Raises
    ------
    UnsupportedBackendError
        A snapshot contained a backend other than cpu/cuda.
    NoProgramsObservedError
        Both caches were empty — likely zero tests collected or the cache was
        wiped at teardown.
    NoSourceFilesObservedError
        Programs were cached but no source files survived (usually a missing
        ``DACE_compiler_build_folder_mode=development``).
    DeterminismError
        One or more programs differed between the two runs. Under
        ``tolerate_missing=True`` this requires at least one *content*
        difference.
    """
    snap1 = snapshot_run(cache1)
    snap2 = snapshot_run(cache2)
    results = compare(snap1, snap2)

    if diffs_dir is not None:
        write_diffs(results, diffs_dir)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(results, tolerate_missing=tolerate_missing))

    if not results:
        raise NoProgramsObservedError(
            "no programs observed in either cache:\n"
            f"  run1 ({cache1}): {_diagnose_empty_cache(cache1)}\n"
            f"  run2 ({cache2}): {_diagnose_empty_cache(cache2)}"
        )

    # Safety net for the silent-false-positive case where both runs cached
    # programs but every program's source tree is empty — typically because
    # DaCe's build_folder_mode is `production` (the gt4py default). Without
    # this, the comparator would see {} == {} for every program and report
    # `deterministic` despite there being nothing to compare.
    total_files = sum(len(s.files) for s in snap1.values()) + sum(
        len(s.files) for s in snap2.values()
    )
    if total_files == 0:
        raise NoSourceFilesObservedError(
            f"{len(results)} program(s) cached, but none of them contain any "
            f"source files under {CODEGEN_SUBDIR}/. This almost always means "
            f"dace's build folder mode is `production` rather than "
            f"`development`, which strips the codegen output after "
            f"compilation. Set DACE_compiler_build_folder_mode=development "
            f"(lowercase matters) before running the tests so the codegen "
            f"survives into the cache."
        )

    # Count true differs (program in both runs, content differs) and missing
    # (program only in one run). Under tolerate_missing, only true differs
    # raise; under strict mode, both do.
    n_true_differs = sum(
        1 for r in results if r.differing_files or (not r.missing_on_one_side and not r.match)
    )
    n_missing = sum(1 for r in results if r.missing_on_one_side)
    n_failed = n_true_differs if tolerate_missing else (n_true_differs + n_missing)

    if n_failed > 0:
        if tolerate_missing:
            msg = (
                f"DaCe codegen is non-deterministic: {n_true_differs}/{len(results)} "
                f"program(s) differ by content (plus {n_missing} cached in only one "
                f"run, ignored under tolerate_missing)"
            )
        else:
            msg = f"DaCe codegen is non-deterministic: {n_failed}/{len(results)} program(s) differ"
        raise DeterminismError(msg, results)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dace_deterministic_codegen",
        description=(
            "Compare two gt4py.next build caches and check whether the DaCe "
            "generated source files are byte-identical between them."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run1",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the first cache root (the .gt4py_cache/ directory).",
    )
    p.add_argument(
        "--run2",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the second cache root. Same conventions as --run1.",
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
    p.add_argument(
        "--tolerate-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to skip programs cached in only one of the two runs. "
            "Default: lenient — only content differences in shared programs "
            "raise. Pass --no-tolerate-missing for strict mode, where any "
            "program absent from one cache also counts as a determinism "
            "failure."
        ),
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
            tolerate_missing=args.tolerate_missing,
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
        print(render_report(e.results, tolerate_missing=args.tolerate_missing))
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(render_report(results, tolerate_missing=args.tolerate_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
