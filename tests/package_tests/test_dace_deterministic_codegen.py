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
    NoProgramsObservedError,
    NoSourceFilesObservedError,
    UnsupportedBackendError,
    check_determinism,
    compare,
    _collect,
    render_report,
)


def _program(cache: pathlib.Path, name: str, salt: str, sources: dict[str, str]) -> None:
    digest = hashlib.sha256(salt.encode()).hexdigest()
    folder = cache / f"{name}_{digest}"
    for rel, content in sources.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_collect_groups_by_logical_name(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _program(cache, "apply_diffusion", "a", {"src/cpu/k.cpp": "x"})
    assert set(_collect(cache)) == {"apply_diffusion"}


def test_collect_ignores_non_program_dirs(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _program(cache, "p", "a", {"src/cpu/k.cpp": "x"})
    (cache / "translation_cache").mkdir(parents=True)
    assert set(_collect(cache)) == {"p"}


def test_collect_unsupported_backend_raises(tmp_path):
    cache = tmp_path / ".gt4py_cache"
    _program(cache, "p", "a", {"src/mpi/k.cpp": "x"})
    with pytest.raises(UnsupportedBackendError, match="mpi"):
        _collect(cache)


def test_deterministic_single_program_matches(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "copy", "a", {"src/cpu/k.cpp": "stable"})
    _program(c2, "copy", "a", {"src/cpu/k.cpp": "stable"})
    (r,) = compare(_collect(c1), _collect(c2))
    assert r.match


def test_nondeterministic_program_pairs_despite_digest(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "abs", "a", {"src/cpu/k.cpp": "A\n"})
    _program(c2, "abs", "b", {"src/cpu/k.cpp": "B\n"})
    (r,) = compare(_collect(c1), _collect(c2))
    assert not r.match
    assert not r.missing_on_one_side
    assert any(rel == "src/cpu/k.cpp" for rel, _ in r.only_in_run1)
    assert any(rel == "src/cpu/k.cpp" for rel, _ in r.only_in_run2)


def test_multiple_programs_same_name_same_bag_matches(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    for salt, src in [("a", "S1"), ("b", "S2"), ("c", "S3")]:
        _program(c1, "impl", salt, {"src/cpu/impl.cpp": src})
    for salt, src in [("x", "S1"), ("y", "S2"), ("z", "S3")]:
        _program(c2, "impl", salt, {"src/cpu/impl.cpp": src})
    (r,) = compare(_collect(c1), _collect(c2))
    assert r.match
    assert r.count1 == 3 and r.count2 == 3


def test_multiple_programs_perturbed_source_differs(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    for salt, src in [("a", "S1"), ("b", "S2")]:
        _program(c1, "solve", salt, {"src/cpu/k.cpp": src})
    for salt, src in [("x", "S1"), ("y", "S2_PERTURBED")]:
        _program(c2, "solve", salt, {"src/cpu/k.cpp": src})
    (r,) = compare(_collect(c1), _collect(c2))
    assert not r.match
    assert not r.missing_on_one_side


def test_count_divergence_differs(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    for salt, src in [("a", "S1"), ("b", "S2"), ("c", "S3")]:
        _program(c1, "impl", salt, {"src/cpu/k.cpp": src})
    for salt, src in [("x", "S1"), ("y", "S2")]:
        _program(c2, "impl", salt, {"src/cpu/k.cpp": src})
    (r,) = compare(_collect(c1), _collect(c2))
    assert not r.match
    assert r.count1 == 3 and r.count2 == 2


def test_missing_on_one_side(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "only1", "a", {"src/cpu/k.cpp": "x"})
    c2.mkdir(parents=True)
    (r,) = compare(_collect(c1), _collect(c2))
    assert r.missing_on_one_side
    assert not r.match


def test_check_determinism_pass(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "copy", "a", {"src/cpu/k.cpp": "stable"})
    _program(c2, "copy", "a", {"src/cpu/k.cpp": "stable"})
    report = tmp_path / "report.txt"
    results = check_determinism(c1, c2, report_path=report)
    assert all(r.match for r in results)
    assert "deterministic" in report.read_text()


def test_check_determinism_differs_raises_and_writes_diff(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "abs", "a", {"src/cpu/k.cpp": "A"})
    _program(c2, "abs", "b", {"src/cpu/k.cpp": "B"})
    diffs = tmp_path / "diffs"
    with pytest.raises(DeterminismError):
        check_determinism(c1, c2, diffs_dir=diffs)
    assert (diffs / "abs.txt").exists()


def test_check_determinism_tolerate_missing(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "shared", "a", {"src/cpu/k.cpp": "x"})
    _program(c2, "shared", "a", {"src/cpu/k.cpp": "x"})
    _program(c1, "only1", "b", {"src/cpu/k.cpp": "y"})
    check_determinism(c1, c2, tolerate_missing=True)
    with pytest.raises(DeterminismError):
        check_determinism(c1, c2, tolerate_missing=False)


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
    text = render_report(compare(_collect(c1), _collect(c2)))
    assert "DIFFER" in text and "NON-DETERMINISTIC" in text
