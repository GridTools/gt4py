#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Tests for the DaCe codegen determinism check."""

from __future__ import annotations

import hashlib
import pathlib
import stat
import sys

import pytest
from dace_determinism import (
    CacheFolderPatternError,
    DeterminismError,
    NoComparableProgramsError,
    NoProgramsObservedError,
    NoSourceFilesObservedError,
    PytestRunError,
    UnsupportedBackendError,
    _compile_folder_pattern,
    _scan,
    check_determinism,
    cli,
    compare,
    fetch_cache_folder_pattern,
    render_report,
    run_determinism_check,
)
from typer.testing import CliRunner


# Mirrors gt4py's current cache folder naming, purely to shape realistic test
# fixtures. The authoritative pattern lives in gt4py (`CACHE_FOLDER_NAME_PATTERN`,
# round-trip tested against `get_cache_folder` there) and the script always
# fetches that one at runtime — it has no built-in copy.
_FOLDER_PATTERN = (
    r"(?P<name>.+)_(?P<fingerprint>[0-9a-f]{16})_(?P<version_id>.+?)"
    r"(?:_(?P<build_context_id>[0-9a-f]{16}))?"
)
_FOLDER_RE = _compile_folder_pattern(_FOLDER_PATTERN)


def _bags(cache: pathlib.Path):
    return _scan(cache, _FOLDER_RE)[0]


def _hex16(salt: str) -> str:
    return hashlib.sha256(salt.encode()).hexdigest()[:16]


def _folder_name(name: str, salt: str) -> str:
    # Mirrors gt4py's current naming: {name}_{fingerprint}_{version_id}_{context_id}.
    return f"{name}_{_hex16(salt)}_1.0.0_{_hex16('build config')}"


def _program(cache: pathlib.Path, name: str, salt: str, sources: dict[str, str]) -> None:
    folder = cache / _folder_name(name, salt)
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
    (cache / _folder_name("p", "a") / "src").mkdir(parents=True)
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
        check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)


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
    check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)  # passes: the comparable name matched


def test_missing_on_one_side_skipped(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c2, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "only1", "a", {"src/cpu/k.cpp": "x"})
    by_name = {r.name: r for r in compare(_bags(c1), _bags(c2))}
    assert by_name["only1"].skipped and not by_name["only1"].match
    check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)


def test_nothing_comparable_raises(tmp_path):
    # Every name has a differing count -> nothing to compare -> not a green pass.
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "a", "1", {"src/cpu/k.cpp": "A1"})
    _program(c1, "a", "2", {"src/cpu/k.cpp": "A2"})
    _program(c2, "a", "1", {"src/cpu/k.cpp": "A1"})
    with pytest.raises(NoComparableProgramsError):
        check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)


def test_one_empty_run_raises_not_green(tmp_path):
    # run2 produced no programs at all: must not be reported as deterministic.
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "p", "a", {"src/cpu/k.cpp": "S1"})
    c2.mkdir(parents=True)
    with pytest.raises(NoComparableProgramsError):
        check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)


def test_check_determinism_pass_writes_report(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "copy", "a", {"src/cpu/k.cpp": "stable"})
    _program(c2, "copy", "a", {"src/cpu/k.cpp": "stable"})
    report = tmp_path / "report.txt"
    results = check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN, report_path=report)
    assert all(r.match for r in results)
    assert "deterministic" in report.read_text()


def test_check_determinism_differs_writes_diff(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "abs", "a", {"src/cpu/k.cpp": "A"})
    _program(c2, "abs", "b", {"src/cpu/k.cpp": "B"})
    diffs = tmp_path / "diffs"
    with pytest.raises(DeterminismError):
        check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN, diffs_dir=diffs)
    body = (diffs / "abs.txt").read_text().splitlines()
    assert body[0] == "abs"
    assert "src/cpu/k.cpp" in body


def test_no_programs_raises(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    c1.mkdir(parents=True)
    c2.mkdir(parents=True)
    with pytest.raises(NoProgramsObservedError):
        check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)


def test_no_source_files_raises(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    (c1 / _folder_name("p", "a")).mkdir(parents=True)
    (c2 / _folder_name("p", "a")).mkdir(parents=True)
    with pytest.raises(NoSourceFilesObservedError, match="development"):
        check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)


def test_report_lists_mismatching_sources(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "p", "a", {"src/cpu/k.cpp": "v1"})
    _program(c2, "p", "b", {"src/cpu/k.cpp": "v2"})
    text = render_report(compare(_bags(c1), _bags(c2)))
    assert "[DIFFER]" in text and "NON-DETERMINISTIC" in text
    assert "only in" not in text and "(run1:" not in text


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


# --- run-health classification of count mismatches ---


def test_count_mismatch_fails_when_runs_healthy(tmp_path):
    # In a clean pair a differing count cannot be a failed test; it is
    # non-deterministic program count and must be a failure.
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c2, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "wobble", "a", {"src/cpu/k.cpp": "W1"})
    _program(c1, "wobble", "b", {"src/cpu/k.cpp": "W2"})
    _program(c2, "wobble", "x", {"src/cpu/k.cpp": "W1"})
    check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN)  # health unknown -> tolerated skip
    check_determinism(
        c1, c2, folder_pattern=_FOLDER_PATTERN, runs_healthy=False
    )  # a test failed -> tolerated skip
    with pytest.raises(DeterminismError):
        check_determinism(
            c1, c2, folder_pattern=_FOLDER_PATTERN, runs_healthy=True
        )  # clean pair -> failure


def test_clean_pair_all_match_passes(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "copy", "a", {"src/cpu/k.cpp": "S1"})
    _program(c2, "copy", "a", {"src/cpu/k.cpp": "S1"})
    check_determinism(c1, c2, folder_pattern=_FOLDER_PATTERN, runs_healthy=True)


def test_report_marks_count_as_failure_when_healthy(tmp_path):
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    _program(c1, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c2, "ok", "a", {"src/cpu/k.cpp": "S1"})
    _program(c1, "wobble", "a", {"src/cpu/k.cpp": "W1"})
    _program(c1, "wobble", "b", {"src/cpu/k.cpp": "W2"})
    _program(c2, "wobble", "x", {"src/cpu/k.cpp": "W1"})
    results = compare(_bags(c1), _bags(c2))
    healthy = render_report(results, runs_healthy=True)
    assert "[COUNT ]" in healthy and "NON-DETERMINISTIC" in healthy
    unknown = render_report(results)
    assert "[SKIP  ]" in unknown and "Re-run" in unknown


# --- orchestration (run_determinism_check) ---
#
# These tests drive the run-twice loop with a stub interpreter instead of a real
# gt4py + pytest run: the stub mimics `python -m pytest ... --junit-xml=PATH` by
# fabricating a `.gt4py_cache` program tree under GT4PY_BUILD_CACHE_DIR plus a
# JUnit XML, so the loop, health parsing, and comparison are exercised without
# needing gt4py installed.
#
# The stub also answers the folder-name pattern probe (`python -c ...`) with its
# OWN naming scheme (no version/context segments, unlike gt4py's). The
# orchestration only finds the stub's program folders through the fetched
# pattern — there is no fallback — making these tests an end-to-end guard on
# the pattern-fetch mechanism.

_STUB_FOLDER_PATTERN = r"(?P<name>.+)_(?P<salt>[0-9a-f]{16})"

_STUB_PYTEST = """\
#!{python}
import hashlib, os, pathlib, sys

if sys.argv[1:2] == ["-c"]:  # the folder-name pattern probe
    print({pattern!r})
    sys.exit(0)

junit = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--junit-xml="))
cache = pathlib.Path(os.environ["GT4PY_BUILD_CACHE_DIR"]) / ".gt4py_cache"
run_name = pathlib.Path(os.environ["GT4PY_BUILD_CACHE_DIR"]).name
# Codegen differs between runs only when perturbation is requested.
content = run_name if os.environ.get("FAKE_PERTURB") else "stable"
folder = cache / ("prog_" + hashlib.sha256(b"prog").hexdigest()[:16]) / "src" / "cpu"
folder.mkdir(parents=True, exist_ok=True)
(folder / "k.cpp").write_text(content)
pathlib.Path(junit).write_text('<testsuite tests="1" failures="0" errors="0" />')
"""


def _stub_interpreter(tmp_path: pathlib.Path) -> str:
    stub = tmp_path / "fake_pytest.py"
    stub.write_text(_STUB_PYTEST.format(python=sys.executable, pattern=_STUB_FOLDER_PATTERN))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return str(stub)


# An explicit `dacecache` keeps these tests hermetic: with the default
# (`<cwd>/.dacecache`) a run started from the repo root would wipe a real cache.


def test_run_determinism_check_deterministic_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_PERTURB", raising=False)
    workdir = tmp_path / "_workdir"
    results = run_determinism_check(
        ["-q"],
        workdir=workdir,
        python=_stub_interpreter(tmp_path),
        dacecache=tmp_path / ".dacecache",
        self_check=False,
    )
    assert results and all(r.match for r in results)
    assert "deterministic" in (workdir / "report.txt").read_text()
    # the per-run caches are reclaimed, the report is kept
    assert not (workdir / "run1").exists() and not (workdir / "run2").exists()


def test_run_determinism_check_detects_nondeterminism(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PERTURB", "1")
    workdir = tmp_path / "_workdir"
    with pytest.raises(DeterminismError):
        run_determinism_check(
            ["-q"],
            workdir=workdir,
            python=_stub_interpreter(tmp_path),
            dacecache=tmp_path / ".dacecache",
            self_check=False,
        )
    report = (workdir / "report.txt").read_text()
    assert "NON-DETERMINISTIC" in report


def test_run_determinism_check_infra_failure_raises(tmp_path):
    # An interpreter that exits non-zero (and != 1/5) on the pytest run is an
    # infrastructure error. (It still answers the pattern probe: an unanswered
    # probe would abort earlier with CacheFolderPatternError instead.)
    stub = tmp_path / "boom.py"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        f"    print({_STUB_FOLDER_PATTERN!r})\n"
        "    sys.exit(0)\n"
        "sys.exit(2)\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    with pytest.raises(PytestRunError, match="unexpected code"):
        run_determinism_check(
            ["-q"],
            workdir=tmp_path / "_workdir",
            python=str(stub),
            dacecache=tmp_path / ".dacecache",
            self_check=False,
        )


def test_run_determinism_check_default_dacecache_tracks_cwd(tmp_path, monkeypatch):
    # DaCe writes .dacecache relative to the process CWD, not relative to the
    # workdir. With the workdir nested below the CWD, the default dacecache must
    # still resolve to <cwd>/.dacecache (and get wiped so the two runs cannot
    # share artifacts) — not <workdir.parent>/.dacecache.
    monkeypatch.delenv("FAKE_PERTURB", raising=False)
    monkeypatch.chdir(tmp_path)
    cwd_dacecache = tmp_path / ".dacecache"
    cwd_dacecache.mkdir()
    (cwd_dacecache / "stale.so").write_text("old build artifact")
    run_determinism_check(  # no `dacecache=` -> exercises the default
        ["-q"],
        workdir=tmp_path / "sub" / "_workdir",
        python=_stub_interpreter(tmp_path),
        self_check=False,
    )
    assert not cwd_dacecache.exists()


def test_run_determinism_check_env_overrides_are_set(tmp_path):
    # The stub records the determinism-relevant env vars it was invoked with.
    stub = tmp_path / "record_env.py"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        f"    print({_STUB_FOLDER_PATTERN!r})\n"
        "    sys.exit(0)\n"
        "junit = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--junit-xml='))\n"
        "cache = pathlib.Path(os.environ['GT4PY_BUILD_CACHE_DIR']) / '.gt4py_cache'\n"
        "rec = cache.parent.parent / (pathlib.Path(os.environ['GT4PY_BUILD_CACHE_DIR']).name + '.env')\n"
        "rec.write_text(os.environ.get('DACE_compiler_build_folder_mode', '') + ',' "
        "+ os.environ.get('GT4PY_BUILD_CACHE_LIFETIME', ''))\n"
        "cache.mkdir(parents=True, exist_ok=True)\n"
        'pathlib.Path(junit).write_text(\'<testsuite tests="0" failures="0" errors="0" />\')\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    workdir = tmp_path / "_workdir"
    with pytest.raises(NoProgramsObservedError):  # empty caches -> nothing to compare
        run_determinism_check(
            ["-q"],
            workdir=workdir,
            python=str(stub),
            dacecache=tmp_path / ".dacecache",
            self_check=False,
        )
    assert (workdir / "run1.env").read_text() == "development,persistent"


# --- CLI `ci-check` exit-code mapping ---
#
# Regression guard: every comparison exception subclasses RuntimeError, so the
# `ci-check` handler must map each to its documented exit code rather than
# collapsing them onto a single catch-all.


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (DeterminismError("nondeterministic codegen", []), 1),
        (NoProgramsObservedError("no programs"), 3),
        (NoComparableProgramsError("nothing comparable"), 2),
        (NoSourceFilesObservedError("no sources"), 2),
        (UnsupportedBackendError("bad backend"), 2),
        (PytestRunError("pytest exited with unexpected code 2"), 2),
        (CacheFolderPatternError("pattern unavailable"), 2),
    ],
)
def test_ci_check_command_maps_exceptions_to_exit_codes(monkeypatch, tmp_path, exc, expected_code):
    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr("dace_determinism.run_determinism_check", boom)
    result = CliRunner().invoke(
        cli, ["ci-check", "--workdir", str(tmp_path / "_wd"), "--no-self-check"]
    )
    assert result.exit_code == expected_code


def test_ci_check_command_success_echoes_report(monkeypatch, tmp_path):
    workdir = tmp_path / "_wd"
    workdir.mkdir()
    (workdir / "report.txt").write_text("RESULT: codegen deterministic — 1 program(s) match.\n")
    monkeypatch.setattr("dace_determinism.run_determinism_check", lambda *a, **k: [])
    result = CliRunner().invoke(cli, ["ci-check", "--workdir", str(workdir), "--no-self-check"])
    assert result.exit_code == 0
    assert "deterministic" in result.output


# --- folder-name pattern handling ---
#
# The folder-name layout belongs to gt4py: `CACHE_FOLDER_NAME_PATTERN` in
# gt4py/next/otf/compilation/cache.py, held in sync with `get_cache_folder` by a
# round-trip test there. These tests cover this script's side of the contract:
# the pattern is always fetched from a gt4py interpreter, an unusable or
# unavailable pattern is a hard error (no fallback), and the `check` subcommand
# uses the fetched pattern.


def test_pattern_without_name_group_rejected(tmp_path):
    with pytest.raises(ValueError, match="name"):
        check_determinism(tmp_path / "r1", tmp_path / "r2", folder_pattern=r".+_[0-9a-f]{16}")


def test_check_cli_uses_fetched_pattern(monkeypatch, tmp_path):
    # The check subcommand reads the pattern from the interpreter running the
    # script; stub the fetch so the test is independent of this environment.
    c1 = tmp_path / "r1" / ".gt4py_cache"
    c2 = tmp_path / "r2" / ".gt4py_cache"
    for cache in (c1, c2):
        src = cache / "copy-ABC" / "src" / "cpu"
        src.mkdir(parents=True)
        (src / "k.cpp").write_text("stable")

    monkeypatch.setattr(
        "dace_determinism.fetch_cache_folder_pattern", lambda python: r"(?P<name>.+)-[A-Z]{3}"
    )
    result = CliRunner().invoke(cli, ["check", "--run1", str(c1), "--run2", str(c2)])
    assert result.exit_code == 0
    assert "deterministic" in result.output


def test_check_cli_fails_without_pattern_source(monkeypatch, tmp_path):
    def boom(python):
        raise CacheFolderPatternError("no gt4py in this interpreter")

    monkeypatch.setattr("dace_determinism.fetch_cache_folder_pattern", boom)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    result = CliRunner().invoke(
        cli, ["check", "--run1", str(tmp_path / "r1"), "--run2", str(tmp_path / "r2")]
    )
    assert result.exit_code == 2


def _fake_interpreter(tmp_path: pathlib.Path, body: str) -> str:
    stub = tmp_path / "fake_python.py"
    stub.write_text(f"#!{sys.executable}\n{body}")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return str(stub)


def test_fetch_pattern_returns_advertised_pattern(tmp_path):
    python = _fake_interpreter(tmp_path, f"print({_STUB_FOLDER_PATTERN!r})\n")
    assert fetch_cache_folder_pattern(python) == _STUB_FOLDER_PATTERN


def test_fetch_pattern_raises_on_failing_interpreter(tmp_path):
    python = _fake_interpreter(tmp_path, "import sys; sys.exit(1)\n")
    with pytest.raises(CacheFolderPatternError, match="could not provide"):
        fetch_cache_folder_pattern(python)


def test_fetch_pattern_raises_on_unusable_pattern(tmp_path):
    for body in ("print('no name group here')\n", "print('(?P<name>unbalanced')\n", "print()\n"):
        python = _fake_interpreter(tmp_path, body)
        with pytest.raises(CacheFolderPatternError, match="unusable"):
            fetch_cache_folder_pattern(python)


def test_run_determinism_check_aborts_without_pattern(tmp_path):
    # No pattern source -> abort before any test run, instead of guessing.
    python = _fake_interpreter(tmp_path, "import sys; sys.exit(1)\n")
    with pytest.raises(CacheFolderPatternError):
        run_determinism_check(
            ["-q"],
            workdir=tmp_path / "_workdir",
            python=python,
            dacecache=tmp_path / ".dacecache",
            self_check=False,
        )
    assert not (tmp_path / "_workdir").exists()  # aborted before creating anything
