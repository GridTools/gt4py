# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the DaCe codegen determinism comparator.

The module under test (``scripts/dace_deterministic_codegen.py``) is
pure-stdlib and operates on directory trees, so these tests build small fake
``.gt4py_cache``-style trees with :func:`tmp_path` and exercise the
snapshot / compare / report / ``check_determinism`` surface directly — no
gt4py or dace runtime needed.
"""

from __future__ import annotations

import pathlib
import sys

import pytest


# The comparator lives in <repo>/scripts/.
def _find_repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "scripts" / "dace_deterministic_codegen.py").is_file():
            return parent
    raise RuntimeError("could not locate the repo root (scripts/dace_deterministic_codegen.py)")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dace_deterministic_codegen import (  # noqa: E402
    DeterminismError,
    NoProgramsObservedError,
    NoSourceFilesObservedError,
    ProgramResult,
    ProgramSnapshot,
    UnsupportedBackendError,
    check_determinism,
    compare,
    render_report,
    snapshot_run,
    write_diffs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_program(
    cache_root: pathlib.Path,
    folder_name: str,
    files: dict[str, str],
) -> None:
    """Create one program folder ``<cache_root>/<folder_name>`` whose ``files``
    map relpaths (e.g. ``src/cpu/foo.cpp``) to text content."""
    for rel, content in files.items():
        _write(cache_root / folder_name / rel, content)


# ---------------------------------------------------------------------------
# snapshot_run
# ---------------------------------------------------------------------------


def test_snapshot_keys_by_full_folder_name_no_collision(tmp_path):
    # Two distinct programs that share an entry-point name but differ by cache
    # digest must NOT collide — keying by the full folder name keeps them apart.
    cache = tmp_path / ".gt4py_cache"
    digest_a = "a" * 64
    digest_b = "b" * 64
    _make_program(cache, f"prog_{digest_a}", {"src/cpu/k.cpp": "AAA"})
    _make_program(cache, f"prog_{digest_b}", {"src/cpu/k.cpp": "BBB"})

    snap = snapshot_run(cache)

    assert set(snap) == {f"prog_{digest_a}", f"prog_{digest_b}"}
    assert snap[f"prog_{digest_a}"].files["src/cpu/k.cpp"].sha256 != (
        snap[f"prog_{digest_b}"].files["src/cpu/k.cpp"].sha256
    )


def test_snapshot_picks_up_hip_under_cuda(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _make_program(
        cache,
        "prog_" + "c" * 64,
        {"src/cuda/k.cu": "kernel", "src/cuda/hip/k.cpp": "hip-kernel"},
    )
    snap = snapshot_run(cache)
    (prog,) = snap.values()
    assert set(prog.files) == {"src/cuda/k.cu", "src/cuda/hip/k.cpp"}


def test_snapshot_unsupported_backend_raises(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _make_program(cache, "prog_" + "d" * 64, {"src/mpi/k.cpp": "x"})
    with pytest.raises(UnsupportedBackendError, match="mpi"):
        snapshot_run(cache)


def test_snapshot_missing_src_is_empty_snapshot(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    (cache / ("prog_" + "e" * 64)).mkdir(parents=True)
    snap = snapshot_run(cache)
    (prog,) = snap.values()
    assert prog.files == {}


def test_snapshot_nonexistent_root_is_empty(tmp_path):
    assert snapshot_run(tmp_path / "does-not-exist") == {}


def test_snapshot_records_absolute_paths(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _make_program(cache, "prog_" + "f" * 64, {"src/cpu/k.cpp": "data"})
    snap = snapshot_run(cache)
    (prog,) = snap.values()
    entry = prog.files["src/cpu/k.cpp"]
    assert entry.abspath.is_absolute()
    assert entry.abspath.read_text() == "data"


# ---------------------------------------------------------------------------
# compare / ProgramResult
# ---------------------------------------------------------------------------


def test_compare_identical_matches(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    for c in (c1, c2):
        _make_program(c, "prog_" + "a" * 64, {"src/cpu/k.cpp": "same"})
    (r,) = compare(snapshot_run(c1), snapshot_run(c2))
    assert r.match
    assert r.present_in_run1 and r.present_in_run2
    assert not r.missing_on_one_side


def test_compare_content_diff(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    _make_program(c1, "prog_" + "a" * 64, {"src/cpu/k.cpp": "v1"})
    _make_program(c2, "prog_" + "a" * 64, {"src/cpu/k.cpp": "v2"})
    (r,) = compare(snapshot_run(c1), snapshot_run(c2))
    assert not r.match
    assert r.differing_files == ["src/cpu/k.cpp"]
    assert not r.missing_on_one_side  # present in BOTH, just differing content


def test_missing_on_one_side_distinguished_from_extra_file(tmp_path):
    # Program present in both runs, but run2 emitted an extra file. This is a
    # one-sided file list, yet it is NOT a missing-program case — the presence
    # flags must keep these distinct.
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    _make_program(c1, "prog_" + "a" * 64, {"src/cpu/k.cpp": "x"})
    _make_program(c2, "prog_" + "a" * 64, {"src/cpu/k.cpp": "x", "src/cpu/extra.cpp": "y"})
    (r,) = compare(snapshot_run(c1), snapshot_run(c2))
    assert r.present_in_run1 and r.present_in_run2
    assert not r.missing_on_one_side
    assert r.only_in_run2 == ["src/cpu/extra.cpp"]
    assert not r.match


def test_missing_program_is_missing_on_one_side(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    _make_program(c1, "prog_" + "a" * 64, {"src/cpu/k.cpp": "x"})
    c2.mkdir(parents=True)  # empty second run
    (r,) = compare(snapshot_run(c1), snapshot_run(c2))
    assert r.present_in_run1 and not r.present_in_run2
    assert r.missing_on_one_side
    assert not r.match


def test_match_is_derived_not_stored():
    # `match` is a computed property; mutating the diff lists flips it without
    # any separate bookkeeping (guards against the fields drifting out of sync).
    snap = ProgramSnapshot(name="p", folder=pathlib.Path("."), files={})
    r = ProgramResult(
        name="p",
        snapshot1=snap,
        snapshot2=snap,
        differing_files=[],
        only_in_run1=[],
        only_in_run2=[],
    )
    assert r.match
    r.differing_files = ["src/cpu/k.cpp"]
    assert not r.match


# ---------------------------------------------------------------------------
# write_diffs
# ---------------------------------------------------------------------------


def test_write_diffs_emits_unified_diff(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    _make_program(c1, "prog_" + "a" * 64, {"src/cpu/k.cpp": "line1\nold\n"})
    _make_program(c2, "prog_" + "a" * 64, {"src/cpu/k.cpp": "line1\nnew\n"})
    results = compare(snapshot_run(c1), snapshot_run(c2))
    diffs_dir = tmp_path / "diffs"
    write_diffs(results, diffs_dir)
    diff_file = diffs_dir / ("prog_" + "a" * 64) / "src__cpu__k.cpp.diff"
    assert diff_file.exists()
    body = diff_file.read_text()
    assert "-old" in body and "+new" in body


# ---------------------------------------------------------------------------
# check_determinism (end to end)
# ---------------------------------------------------------------------------


def test_check_determinism_pass(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    for c in (c1, c2):
        _make_program(c, "prog_" + "a" * 64, {"src/cpu/k.cpp": "stable"})
    report = tmp_path / "report.txt"
    results = check_determinism(c1, c2, report_path=report)
    assert all(r.match for r in results)
    assert "deterministic" in report.read_text()


def test_check_determinism_content_diff_raises(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    _make_program(c1, "prog_" + "a" * 64, {"src/cpu/k.cpp": "v1"})
    _make_program(c2, "prog_" + "a" * 64, {"src/cpu/k.cpp": "v2"})
    with pytest.raises(DeterminismError):
        check_determinism(c1, c2, diffs_dir=tmp_path / "diffs")


def test_check_determinism_tolerate_missing(tmp_path):
    # A program present in only one run is tolerated by default but fails
    # under strict mode.
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    _make_program(c1, "prog_" + "a" * 64, {"src/cpu/k.cpp": "x"})
    _make_program(c1, "prog_" + "b" * 64, {"src/cpu/k.cpp": "y"})  # only in run1
    _make_program(c2, "prog_" + "a" * 64, {"src/cpu/k.cpp": "x"})

    # lenient (default): the shared program matches -> no raise
    check_determinism(c1, c2, tolerate_missing=True)

    # strict: the one-sided program now counts as a failure
    with pytest.raises(DeterminismError):
        check_determinism(c1, c2, tolerate_missing=False)


def test_check_determinism_no_programs_raises(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    c1.mkdir(parents=True)
    c2.mkdir(parents=True)
    with pytest.raises(NoProgramsObservedError):
        check_determinism(c1, c2)


def test_check_determinism_no_source_files_raises(tmp_path):
    # Program folders exist but carry no src/ — mimics dace `production` mode
    # stripping the codegen. Must fail loudly rather than report "deterministic".
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    (c1 / ("prog_" + "a" * 64)).mkdir(parents=True)
    (c2 / ("prog_" + "a" * 64)).mkdir(parents=True)
    with pytest.raises(NoSourceFilesObservedError, match="development"):
        check_determinism(c1, c2)


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def test_render_report_flags_nondeterminism(tmp_path):
    c1 = tmp_path / "run1" / ".gt4py_cache"
    c2 = tmp_path / "run2" / ".gt4py_cache"
    _make_program(c1, "prog_" + "a" * 64, {"src/cpu/k.cpp": "v1"})
    _make_program(c2, "prog_" + "a" * 64, {"src/cpu/k.cpp": "v2"})
    results = compare(snapshot_run(c1), snapshot_run(c2))
    text = render_report(results)
    assert "NON-DETERMINISTIC" in text
    assert "DIFFER" in text
