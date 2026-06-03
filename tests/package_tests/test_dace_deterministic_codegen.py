# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DaCe codegen determinism check."""

from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest


def _repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "scripts" / "dace_deterministic_codegen.py").is_file():
            return parent
    raise RuntimeError("scripts/dace_deterministic_codegen.py not found")


sys.path.insert(0, str(_repo_root()))

from scripts.dace_deterministic_codegen import (  # noqa: E402
    DeterminismError,
    NoComparableProgramsError,
    NoProgramsObservedError,
    NoSourceFilesObservedError,
    UnsupportedBackendError,
    check_determinism,
    compare,
    _scan,
    render_report,
)


def _bags(cache: pathlib.Path):
    return _scan(cache)[0]


def _program(cache: pathlib.Path, name: str, salt: str, sources: dict[str, str]) -> None:
    folder = cache / f"{name}_{hashlib.sha256(salt.encode()).hexdigest()}"
    for rel, content in sources.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_collect_groups_by_logical_name(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _program(cache, "apply_diffusion", "a", {"src/cpu/k.cpp": "x"})
    assert set(_bags(cache)) == {"apply_diffusion"}


def test_collect_ignores_non_program_dirs(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _program(cache, "p", "a", {"src/cpu/k.cpp": "x"})
    (cache / "translation_cache").mkdir(parents=True)
    assert set(_bags(cache)) == {"p"}


def test_collect_unsupported_backend_raises(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _program(cache, "p", "a", {"src/mpi/k.cpp": "x"})
    with pytest.raises(UnsupportedBackendError, match="mpi"):
        _bags(cache)


def test_empty_src_program_excluded(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    (cache / f"p_{hashlib.sha256(b'a').hexdigest()}" / "src").mkdir(parents=True)
    assert "p" not in _bags(cache)


def test_single_program_matches(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "copy", "a", {"src/cpu/k.cpp": "stable"})
    _program(c2, "copy", "a", {"src/cpu/k.cpp": "stable"})
    (r,) = compare(_bags(c1), _bags(c2))
    assert r.match and r.comparable


def test_single_program_differs(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "abs", "a", {"src/cpu/k.cpp": "A\n"})
    _program(c2, "abs", "b", {"src/cpu/k.cpp": "B\n"})
    (r,) = compare(_bags(c1), _bags(c2))
    assert r.differs and not r.match
    assert any(rel == "src/cpu/k.cpp" for rel, _ in r.only_in_run1)
    assert any(rel == "src/cpu/k.cpp" for rel, _ in r.only_in_run2)


def test_multiple_programs_same_multiset_matches(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    for salt, src in [("a", "S1"), ("b", "S2"), ("c", "S3")]:
        _program(c1, "impl", salt, {"src/cpu/impl.cpp": src})
    for salt, src in [("x", "S1"), ("y", "S2"), ("z", "S3")]:
        _program(c2, "impl", salt, {"src/cpu/impl.cpp": src})
    (r,) = compare(_bags(c1), _bags(c2))
    assert r.match and r.count1 == 3 and r.count2 == 3


def test_multiple_programs_reordered_matches(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "k", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "k", "b", {"src/cpu/k.cpp": "S2"})
    _program(c2, "k", "x", {"src/cpu/k.cpp": "S2"})
    _program(c2, "k", "y", {"src/cpu/k.cpp": "S1"})
    (r,) = compare(_bags(c1), _bags(c2))
    assert r.match


def test_equal_count_source_mismatch_differs(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "solve", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "solve", "b", {"src/cpu/k.cpp": "S2"})
    _program(c2, "solve", "x", {"src/cpu/k.cpp": "S1"})
    _program(c2, "solve", "y", {"src/cpu/k.cpp": "S2_PERTURBED"})
    (r,) = compare(_bags(c1), _bags(c2))
    assert r.comparable and r.differs
    with pytest.raises(DeterminismError):
        check_determinism(c1, c2)


def test_count_mismatch_skipped_but_others_compared(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    # a matching name keeps the comparison non-vacuous ...
    _program(c1, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c2, "ok", "a", {"src/cpu/k.cpp": "S1"})
    # ... while this name has a differing count and is skipped.
    _program(c1, "flaky", "a", {"src/cpu/k.cpp": "F1"})
    _program(c1, "flaky", "b", {"src/cpu/k.cpp": "F2"})
    _program(c2, "flaky", "x", {"src/cpu/k.cpp": "F1"})
    by_name = {r.name: r for r in compare(_bags(c1), _bags(c2))}
    assert by_name["ok"].match
    assert by_name["flaky"].skipped and not by_name["flaky"].match
    check_determinism(c1, c2)  # passes: the comparable name matched


def test_missing_on_one_side_skipped(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c2, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "only1", "a", {"src/cpu/k.cpp": "x"})
    by_name = {r.name: r for r in compare(_bags(c1), _bags(c2))}
    assert by_name["only1"].skipped and not by_name["only1"].match
    check_determinism(c1, c2)


def test_nothing_comparable_raises(tmp_path):
    # Every name has a differing count -> nothing to compare -> not a green pass.
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "a", "1", {"src/cpu/k.cpp": "A1"})
    _program(c1, "a", "2", {"src/cpu/k.cpp": "A2"})
    _program(c2, "a", "1", {"src/cpu/k.cpp": "A1"})
    with pytest.raises(NoComparableProgramsError):
        check_determinism(c1, c2)


def test_one_empty_run_raises_not_green(tmp_path):
    # run2 produced no programs at all: must not be reported as deterministic.
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "p", "a", {"src/cpu/k.cpp": "S1"})
    c2.mkdir(parents=True)
    with pytest.raises(NoComparableProgramsError):
        check_determinism(c1, c2)


def test_check_determinism_pass_writes_report(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "copy", "a", {"src/cpu/k.cpp": "stable"})
    _program(c2, "copy", "a", {"src/cpu/k.cpp": "stable"})
    report = tmp_path / "report.txt"
    results = check_determinism(c1, c2, report_path=report)
    assert all(r.match for r in results)
    assert "deterministic" in report.read_text()


def test_check_determinism_differs_writes_diff(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "abs", "a", {"src/cpu/k.cpp": "A"})
    _program(c2, "abs", "b", {"src/cpu/k.cpp": "B"})
    diffs = tmp_path / "diffs"
    with pytest.raises(DeterminismError):
        check_determinism(c1, c2, diffs_dir=diffs)
    assert (diffs / "abs.txt").exists()


def test_no_programs_raises(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    c1.mkdir(parents=True)
    c2.mkdir(parents=True)
    with pytest.raises(NoProgramsObservedError):
        check_determinism(c1, c2)


def test_no_source_files_raises(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    digest = hashlib.sha256(b"a").hexdigest()
    (c1 / f"p_{digest}").mkdir(parents=True)
    (c2 / f"p_{digest}").mkdir(parents=True)
    with pytest.raises(NoSourceFilesObservedError, match="development"):
        check_determinism(c1, c2)


def test_report_lists_mismatching_sources(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "p", "a", {"src/cpu/k.cpp": "v1"})
    _program(c2, "p", "b", {"src/cpu/k.cpp": "v2"})
    text = render_report(compare(_bags(c1), _bags(c2)))
    assert "DIFFER" in text and "NON-DETERMINISTIC" in text


def test_report_notes_skipped(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c2, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "p", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "p", "b", {"src/cpu/k.cpp": "S2"})
    _program(c2, "p", "x", {"src/cpu/k.cpp": "S1"})
    text = render_report(compare(_bags(c1), _bags(c2)))
    assert "SKIP" in text and "Re-run" in text
