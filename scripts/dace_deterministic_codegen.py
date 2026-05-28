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

Supports both gt4py cache layouts:

* ``layout="next"`` (default) — the ``gt4py.next`` cache, a flat
  ``<cache_root>/<name>_<sha256>/src/{cpu,cuda}/...`` structure
  written via ``GT4PY_BUILD_CACHE_DIR``. Compares everything dace
  writes as generated source under each program's ``src/``.
  Unknown top-level backends (anything other than cpu/cuda, with
  HIP nesting under cuda/hip) raise :class:`UnsupportedBackendError`.

* ``layout="cartesian"`` — the ``gt4py.cartesian`` cache, a deeply
  nested ``<cache_root>/py<pyver>_<cachever>/<backend>/<test.module
  .path>/<Class>_<backend>_<id>/...`` structure written via
  ``GT_CACHE_ROOT`` + ``GT_CACHE_PYTEST_DIR`` (with the conftest's
  ``--keep-gtcache`` flag needed to survive ``pytest_sessionfinish``).
  Compares the top-level ``m_*.py`` loader plus ``bindings.{cpp,cu}``
  and ``computation.hpp`` under ``m_*_pyext_BUILD/``. Skips compiled
  artifacts (``*.so``, ``*.o``, ``__pycache__/``), gzipped SDFG
  archives (``*.sdfgz`` — gzip headers carry timestamps), the
  metadata file (``*.cacheinfo``), and the recursive build mirror
  directories (``_GT_/``, ``tmp/``) inside ``_pyext_BUILD/``.

As a library
------------

::

    from scripts.dace_deterministic_codegen import check_determinism

    check_determinism(
        cache1=Path(".../run1/.gt4py_cache"),
        cache2=Path(".../run2/.gt4py_cache"),
        layout="next",                       # or "cartesian"
        diffs_dir=Path(".../diffs"),         # optional
        report_path=Path(".../report.txt"),  # optional
    )

Raises ``DeterminismError`` on mismatch, ``NoProgramsObservedError``
if both caches are empty, ``NoSourceFilesObservedError`` if programs
were cached but contain no source files (typically a missing
``DACE_compiler_build_folder_mode=development``), or
``UnsupportedBackendError`` if the next-layout codegen produced an
unfamiliar top-level backend.

As a CLI
--------

::

    python scripts/dace_deterministic_codegen.py \\
        --run1 path/to/cache1 \\
        --run2 path/to/cache2 \\
        --layout {next,cartesian} \\
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
from typing import Literal


#: Cache layout dispatch tag. ``"next"`` is the gt4py.next cache
#: (flat ``<root>/<name>_<sha256>/src/...`` structure); ``"cartesian"``
#: is the gt4py.cartesian cache (deeply nested ``<root>/py<pyver>_*/
#: <backend>/<test.module.path>/<Class>_<backend>_<id>/...``).
Layout = Literal["next", "cartesian"]


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


# Cartesian layout constants ------------------------------------------------

#: Suffix that marks the per-stencil build directory inside a cartesian
#: program folder, e.g. ``m_TestCopy_dacecpu_4__dacecpu_a8441f26b4_pyext_BUILD/``.
#: Inside that directory we look at the TOP LEVEL only — its ``_GT_/`` and
#: ``tmp/`` subdirectories contain recursive copies of the build path that
#: setuptools spawns when building into an absolute prefix, and those are
#: build artifacts, not codegen output.
CARTESIAN_BUILD_DIR_SUFFIX = "_pyext_BUILD"

#: Names of files inside ``m_*_pyext_BUILD/`` whose contents we byte-compare.
#: ``bindings.{cpp,cu}`` is gt4py.cartesian's pybind11 wrapper around the
#: dace SDFG; ``computation.hpp`` is dace's generated kernel implementation.
#: Both reflect the codegen surface directly — a non-deterministic codegen
#: pass will show up here.
CARTESIAN_BUILD_SOURCE_NAMES: frozenset[str] = frozenset(
    {"bindings.cpp", "bindings.cu", "computation.hpp"}
)

#: Directory-name prefixes inside a program folder that we MUST NOT descend
#: into when searching for ``m_*.py`` loader stubs. ``__pycache__`` is
#: Python's bytecode cache; the build dir holds compiler-generated artifacts.
CARTESIAN_SKIP_DIRS: frozenset[str] = frozenset({"__pycache__"})

#: The 10-hex codegen digest gt4py.cartesian embeds in filenames like
#: ``m_<Class>_<backend>_<id>__<backend>_<DIGEST>.py`` and the build
#: directory ``m_..._<DIGEST>_pyext_BUILD/``. We replace it with the
#: literal ``<DIGEST>`` in the snapshot's relpath keys so that
#: ``bindings.cpp`` from run1 (digest ``a8441f26b4``) and from run2
#: (digest ``bbbbbbbbbb``) map to the same path, surfacing a real
#: content diff rather than two "only-in-one-run" entries that look
#: like flaky test selection.
#:
#: The pattern matches ``_`` + 10 lowercase hex + a boundary that is
#: either a file extension (``.py``, ``.so``, ``.sdfgz``, ``.cacheinfo``)
#: or the literal ``_pyext_BUILD`` suffix. Anchoring on those endings
#: avoids false-positive matches inside arbitrary identifiers.
CARTESIAN_DIGEST_RE = re.compile(r"_(?P<digest>[0-9a-f]{10})(?=(\.|_pyext_BUILD))")

#: When the cartesian determinism tests run under pytest-xdist, each worker
#: is given its own cache root (``<run>/gt_cache/<worker_id>/...``) so that
#: two workers compiling byte-identical SDFGs never share — and never
#: ``rmtree`` out from under each other — the dace build folder named by the
#: SDFG hash. See tests/cartesian_tests/conftest.py:_isolate_dace_cache_per_worker.
#:
#: That worker segment is a *scheduling* artifact, not part of a program's
#: identity: the same stencil may compile on ``gw3`` in run1 and ``gw7`` in
#: run2. If it leaked into ``program_id`` (which is the relative path from
#: the cache root), run1 and run2 would key the same program differently and
#: every program would show as "only in one run". We therefore strip a
#: leading ``gw<N>`` / ``master`` segment from the program path before using
#: it as the comparison key. Serial runs (-n 0) use ``master`` and are
#: handled by the same rule, so the key is identical whether the suite ran
#: serially or in parallel.
CARTESIAN_WORKER_SEG_RE = re.compile(r"^(gw\d+|master)/")


def _strip_worker_segment(program_id: str) -> str:
    """Drop a leading pytest-xdist worker-id path segment if present.

    ``"gw3/py310_1013/dacegpu/.../copy_stencil"`` -> ``"py310_1013/dacegpu/.../copy_stencil"``.
    Idempotent; leaves paths without a worker segment unchanged (so caches
    produced without per-worker isolation still compare correctly).
    """
    return CARTESIAN_WORKER_SEG_RE.sub("", program_id)


def _normalize_cartesian_relpath(relpath: str) -> str:
    """Replace the 10-hex codegen digest in a cartesian relpath with the
    literal token ``<DIGEST>``. Idempotent. Leaves non-matching paths
    unchanged."""
    return CARTESIAN_DIGEST_RE.sub("_<DIGEST>", relpath)


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


@dataclasses.dataclass
class IntraRunDivergence:
    """Record of a program compiled more than once *within a single run*
    (by different xdist workers, under per-worker cache isolation) whose
    copies are NOT byte-identical. This is intra-run non-determinism — a
    strictly stronger finding than the run1-vs-run2 check, since it shows
    codegen varies at fixed inputs within one run. Recorded and reported
    rather than raised: DaCe codegen is known to be non-deterministic, so
    this is an expected finding to surface, not a reason to abort."""

    name: str
    folder_a: Path
    folder_b: Path
    differing_files: list[str]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FileEntry:
    #: Logical relative path used as the comparison key. For ``layout="next"``
    #: this is the on-disk path under the program folder verbatim. For
    #: ``layout="cartesian"`` the 10-hex codegen digest in filenames is
    #: replaced with the literal token ``<DIGEST>`` so equivalent files
    #: across two runs (which carry different digests) still pair up.
    relpath: str
    sha256: str
    #: The actual filename on disk relative to the program folder. Equal to
    #: ``relpath`` for next; un-normalized (real digest preserved) for
    #: cartesian. Used by :func:`write_diffs` to read the file back.
    disk_relpath: str


@dataclasses.dataclass
class ProgramSnapshot:
    name: str
    folder: Path
    files: dict[str, FileEntry]


def snapshot_run(
    cache_root: Path,
    *,
    layout: Layout = "next",
    intra_run_divergences: list["IntraRunDivergence"] | None = None,
) -> dict[str, ProgramSnapshot]:
    """Walk a gt4py build cache and snapshot every program's generated source.

    Dispatches on ``layout`` to either :func:`_snapshot_run_next` (the flat
    ``<root>/<name>_<sha256>/src/...`` structure of gt4py.next) or
    :func:`_snapshot_run_cartesian` (the deeply nested
    ``<root>/py<pyver>_*/<backend>/<test.module.path>/<Class>_<backend>_<id>/...``
    structure of gt4py.cartesian).

    Returns an empty dict (rather than raising) when the path doesn't
    exist or contains no programs in the expected layout; callers can
    pair the empty result with :func:`_diagnose_empty_cache` for a
    human-readable explanation of why.
    """
    if layout == "next":
        return _snapshot_run_next(cache_root)
    if layout == "cartesian":
        return _snapshot_run_cartesian(cache_root, intra_run_divergences=intra_run_divergences)
    raise ValueError(f"unknown layout: {layout!r}, expected 'next' or 'cartesian'")


def _snapshot_run_next(cache_root: Path) -> dict[str, ProgramSnapshot]:
    """Snapshot a gt4py.next-layout cache.

    The input directory's name is irrelevant — the function looks for
    immediate subdirectories matching ``<name>_<64-char-hex-digest>``
    (gt4py.next's program-folder naming) and reads ``<program>/src/``
    recursively under each one. HIP files at ``src/cuda/hip/`` are
    picked up automatically by the recursive walk.

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
            files[rel] = FileEntry(relpath=rel, sha256=_sha256(fpath), disk_relpath=rel)
        out[name] = ProgramSnapshot(name=name, folder=folder, files=files)
    return out


def _snapshot_run_cartesian(
    cache_root: Path,
    *,
    intra_run_divergences: list["IntraRunDivergence"] | None = None,
) -> dict[str, ProgramSnapshot]:
    """Snapshot a gt4py.cartesian-layout cache.

    Program identity is the **relative path** from ``cache_root`` to the
    ``<Class>_<backend>_<id>`` folder, e.g.
    ``py310_1013/dacecpu/cartesian_tests/integration_tests/multi_feature_tests
    /test_suites/TestCopy_dacecpu_4``. Two runs of the same parametrized
    test should produce the same relative path, so this works as a stable
    matching key across runs.

    Files compared per program (everything else is skipped — see module
    docstring for rationale):

      * ``m_*.py`` at the top of the program folder — the gt4py loader
        stub. Its filename embeds the 10-hex codegen digest, and the file
        body references it; either changing is a determinism signal.
      * Files exactly one level inside ``m_*_pyext_BUILD/`` whose basename
        is in :data:`CARTESIAN_BUILD_SOURCE_NAMES` (``bindings.cpp``,
        ``bindings.cu``, ``computation.hpp``). The ``_GT_/`` and ``tmp/``
        subdirectories of the build dir are recursive build-path mirrors
        that setuptools creates when targeting an absolute prefix — they
        contain object files and duplicated outputs, not codegen.
    """
    if not cache_root.is_dir():
        return {}

    # Discover program folders by finding every top-level `m_*.py` loader.
    # "Top-level" here means: not under __pycache__ and not under any
    # *_pyext_BUILD directory (which contains its own copies of generated
    # files we don't want).
    program_dirs: set[Path] = set()
    for py in cache_root.rglob("m_*.py"):
        if not py.is_file():
            continue
        parts = py.relative_to(cache_root).parts
        # Reject if any ancestor is __pycache__ or any *_pyext_BUILD dir
        if any(
            p in CARTESIAN_SKIP_DIRS or p.endswith(CARTESIAN_BUILD_DIR_SUFFIX) for p in parts[:-1]
        ):
            continue
        program_dirs.add(py.parent)

    out: dict[str, ProgramSnapshot] = {}
    for prog_dir in sorted(program_dirs):
        # Strip any leading xdist worker-id segment (gw0/, master/, ...) so a
        # program keyed identically across run1/run2 even if it compiled on a
        # different worker each time. See CARTESIAN_WORKER_SEG_RE.
        program_id = _strip_worker_segment(prog_dir.relative_to(cache_root).as_posix())

        files: dict[str, FileEntry] = {}

        # Top-level m_*.py file(s) — the gt4py loader stub(s).
        for f in sorted(prog_dir.glob("m_*.py")):
            if f.is_file():
                rel = _normalize_cartesian_relpath(f.name)
                files[rel] = FileEntry(relpath=rel, sha256=_sha256(f), disk_relpath=f.name)

        # Files DIRECTLY under any m_*_pyext_BUILD/ — iterdir, not rglob,
        # so we don't descend into _GT_/ or tmp/ which carry build artifacts.
        for build_dir in sorted(prog_dir.glob(f"m_*{CARTESIAN_BUILD_DIR_SUFFIX}")):
            if not build_dir.is_dir():
                continue
            for f in sorted(build_dir.iterdir()):
                if f.is_file() and f.name in CARTESIAN_BUILD_SOURCE_NAMES:
                    disk_rel = f"{build_dir.name}/{f.name}"
                    rel = _normalize_cartesian_relpath(disk_rel)
                    files[rel] = FileEntry(relpath=rel, sha256=_sha256(f), disk_relpath=disk_rel)

        snapshot = ProgramSnapshot(name=program_id, folder=prog_dir, files=files)

        # Under per-worker cache isolation the SAME program can be compiled by
        # more than one xdist worker (e.g. two tests that build an identical
        # stencil land on different workers). After stripping the worker
        # segment they collide on one program_id. That's expected and benign
        # *iff* the two copies are byte-identical — which is what determinism
        # requires. If they differ, codegen is non-deterministic even within a
        # single run (a strictly stronger signal than the run1-vs-run2 check).
        # We RECORD that divergence (rather than raising) so it lands in the
        # determinism report alongside the run1-vs-run2 findings; DaCe codegen
        # is known to be non-deterministic, so an intra-run divergence is an
        # expected finding to report, not a reason to abort the run.
        prev = out.get(program_id)
        if prev is not None:
            prev_hashes = {r: e.sha256 for r, e in prev.files.items()}
            this_hashes = {r: e.sha256 for r, e in snapshot.files.items()}
            if prev_hashes != this_hashes and intra_run_divergences is not None:
                differing = sorted(
                    rel
                    for rel in (set(prev_hashes) | set(this_hashes))
                    if prev_hashes.get(rel) != this_hashes.get(rel)
                )
                intra_run_divergences.append(
                    IntraRunDivergence(
                        name=program_id,
                        folder_a=prev.folder,
                        folder_b=snapshot.folder,
                        differing_files=differing,
                    )
                )
            # Keep the first copy as the canonical snapshot for the
            # run1-vs-run2 comparison; the divergence (if any) is recorded
            # separately above.
            continue
        out[program_id] = snapshot
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _diagnose_empty_cache(cache_root: Path, *, layout: Layout = "next") -> str:
    """Return a one-line explanation of why :func:`snapshot_run` found nothing."""
    if not cache_root.exists():
        return "path does not exist"
    if not cache_root.is_dir():
        return "path exists but is not a directory"

    if layout == "cartesian":
        # For cartesian we expect <root>/py<pyver>_<cachever>/<backend>/...
        # If there's no py*_*/, the user likely passed the wrong path
        # (e.g. the run_dir instead of run_dir/gt_cache).
        subdirs = sorted(p for p in cache_root.iterdir() if p.is_dir())
        if not subdirs:
            return "directory has no subdirectories"
        pyver_dirs = [p for p in subdirs if re.match(r"^py\d+_\d+$", p.name)]
        if not pyver_dirs:
            sample_names = [p.name for p in subdirs[:3]]
            suffix = f" (and {len(subdirs) - 3} more)" if len(subdirs) > 3 else ""
            return (
                f"directory contains {len(subdirs)} subdirectory(ies) but none "
                f"match cartesian's per-Python-version pattern `py<N>_<N>/` "
                f"(saw: {sample_names}{suffix}). Did you pass the cache root, "
                f"or its parent?"
            )
        # py*/ exists but no m_*.py loader stubs were found anywhere
        return (
            "cartesian cache structure present but contains no `m_*.py` loader "
            "stubs at any depth — pytest probably collected zero tests, or the "
            "conftest's `--keep-gtcache` flag wasn't passed and the cache was "
            "wiped at session teardown."
        )

    # Fall through to the next-layout diagnostic
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

    @property
    def missing_on_one_side(self) -> bool:
        """True iff the program was cached in only one of the two runs.

        Distinguished from 'differs by content' (where the program is in
        both runs but at least one file's bytes differ) — the latter is
        always a determinism failure, the former is often a flaky-test
        artifact and can be tolerated via ``tolerate_missing``.

        The compare() loop populates only_in_runN exhaustively with the
        absent side's file list ONLY when the whole program is missing;
        when both programs are present but one happens to carry an extra
        file, only_in_runN contains only that extra file. We distinguish
        the two by requiring exactly one side to be wholly empty (which
        is what compare() emits for the missing-program case).
        """
        return (
            not self.match
            and not self.differing_files
            and (bool(self.only_in_run1) ^ bool(self.only_in_run2))
        )


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
            # `rel` is the canonical (normalized) key; the on-disk filename
            # may differ from it (cartesian normalizes the 10-hex digest).
            # Look up the per-side FileEntry to recover the real path.
            e1 = s1.files.get(rel) if s1 else None
            e2 = s2.files.get(rel) if s2 else None
            f1 = (s1.folder / e1.disk_relpath) if (s1 and e1) else None
            f2 = (s2.folder / e2.disk_relpath) if (s2 and e2) else None
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


def render_report(
    results: list[ProgramResult],
    *,
    tolerate_missing: bool = False,
    intra_run_divergences: list[IntraRunDivergence] | None = None,
) -> str:
    n_total = len(results)
    n_missing = sum(1 for r in results if r.missing_on_one_side)
    n_diff_content = sum(
        1 for r in results if r.differing_files or (not r.match and not r.missing_on_one_side)
    )
    n_match = n_total - n_missing - n_diff_content

    n_intra = len(intra_run_divergences) if intra_run_divergences else 0
    header = (
        f"Programs: {n_total}    matches: {n_match}    "
        f"differs: {n_diff_content}    only-in-one-run: {n_missing}"
        + (f"    intra-run-divergent: {n_intra}" if n_intra else "")
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

    # Intra-run divergences: the same program compiled twice WITHIN one run
    # (by different xdist workers) with differing output. Reported separately
    # because it's a distinct, stronger axis of non-determinism than the
    # run1-vs-run2 comparison above. Informational — does not change the
    # run1-vs-run2 verdict below — but each entry is concrete proof that
    # codegen is non-deterministic at fixed inputs.
    if n_intra:
        lines.append("")
        lines.append(
            f"Intra-run divergences (same program, differing output within ONE run): {n_intra}"
        )
        for d in intra_run_divergences:  # type: ignore[union-attr]
            lines.append(f"  [INTRA ] {d.name}")
            lines.extend(f"           differs: {rel}" for rel in d.differing_files)
            lines.append(f"           copy A: {d.folder_a}")
            lines.append(f"           copy B: {d.folder_b}")

    lines.append("")
    if n_total == 0:
        lines.append("RESULT: no programs observed (nothing was cached).")
    elif n_diff_content == 0 and n_missing == 0:
        base = f"RESULT: codegen deterministic — {n_match} program(s) match."
        lines.append(base)
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
    if n_intra:
        lines.append(
            f"NOTE: additionally, {n_intra} program(s) showed intra-run divergence "
            f"(differing output across xdist workers within a single run)."
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
    layout: Layout = "next",
) -> list[ProgramResult]:
    """Compare two gt4py caches; write artifacts; raise on mismatch.

    Snapshots both caches (using the ``layout``-specific walker) and
    diffs them. Optionally writes per-file unified diffs to
    ``diffs_dir/<program>/`` and a human-readable summary to
    ``report_path``.

    Returns the list of :class:`ProgramResult` on a successful match.

    Parameters
    ----------
    cache1, cache2
        Roots of the two caches to compare. For ``layout="next"``, this
        is the ``.gt4py_cache/`` directory (i.e. the parent of all the
        ``<name>_<sha256>/`` program folders). For ``layout="cartesian"``,
        this is the directory pointed to by ``GT_CACHE_PYTEST_DIR`` (i.e.
        the parent of ``py<pyver>_<cachever>/``).
    layout
        Which cache layout to expect. See module docstring for details.
    tolerate_missing
        See module docstring. Default ``True`` (lenient).
    diffs_dir, report_path
        If set, persist diagnostic artifacts.

    Raises:
        UnsupportedBackendError:
            (next layout only) A snapshot contained a backend other than cpu/cuda.
        NoProgramsObservedError:
            Both caches were empty — likely zero tests collected or the
            cache was wiped at teardown.
        NoSourceFilesObservedError:
            Programs were cached but no source files survived (usually a
            missing ``DACE_compiler_build_folder_mode=development``).
        DeterminismError:
            One or more programs differed between the two runs. Under
            ``tolerate_missing=True`` this requires at least one
            *content* difference.
    """
    intra_run_divergences: list[IntraRunDivergence] = []
    snap1 = snapshot_run(cache1, layout=layout, intra_run_divergences=intra_run_divergences)
    snap2 = snapshot_run(cache2, layout=layout, intra_run_divergences=intra_run_divergences)
    results = compare(snap1, snap2)

    if diffs_dir is not None:
        write_diffs(results, snap1, snap2, diffs_dir)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_report(
                results,
                tolerate_missing=tolerate_missing,
                intra_run_divergences=intra_run_divergences,
            )
        )

    if not results:
        diag1 = _diagnose_empty_cache(cache1, layout=layout)
        diag2 = _diagnose_empty_cache(cache2, layout=layout)
        raise NoProgramsObservedError(
            "no programs observed in either cache:\n"
            f"  run1 ({cache1}): {diag1}\n"
            f"  run2 ({cache2}): {diag2}"
        )

    # Safety net for the silent-false-positive case where both runs cached
    # programs but every program's source tree is empty — typically because
    # dace's build_folder_mode is `production` (the gt4py default). Without
    # this, the comparator would see {} == {} for every program and report
    # `deterministic` despite there being nothing to compare.
    total_files = sum(len(s.files) for s in snap1.values()) + sum(
        len(s.files) for s in snap2.values()
    )
    if total_files == 0:
        if layout == "cartesian":
            hint = (
                "programs were cached but contain none of `m_*.py`, "
                "`bindings.{cpp,cu}`, or `computation.hpp`"
            )
        else:
            hint = "none of them contain any source files under src/"
        raise NoSourceFilesObservedError(
            f"{len(results)} program(s) cached, but {hint}. This almost "
            f"always means dace's build folder mode is `production` rather "
            f"than `development`, which strips the codegen output after "
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
        help=(
            "Path to the first cache root. For --layout next, this is the "
            ".gt4py_cache/ directory. For --layout cartesian, this is the "
            "directory that GT_CACHE_PYTEST_DIR pointed to (the parent of "
            "py<ver>_<format>/)."
        ),
    )
    p.add_argument(
        "--run2",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the second cache root. Same conventions as --run1.",
    )
    p.add_argument(
        "--layout",
        choices=["next", "cartesian"],
        default="next",
        help=(
            "Cache layout. `next` is gt4py.next's flat "
            "<root>/<name>_<sha256>/src/... structure (default). `cartesian` "
            "is gt4py.cartesian's deeply nested "
            "<root>/py<ver>_<format>/<backend>/<test.path>/<Class>_<bk>_<id>/ "
            "structure."
        ),
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
            layout=args.layout,
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
