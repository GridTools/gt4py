#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that gt4py's DaCe backend generates identical source across two runs.

gt4py.next caches each compiled program at ``<root>/<name>_<sha256>/src/...``,
where the digest is derived from the generated source. Non-deterministic codegen
therefore changes the digest from one run to the next. Programs are compared by
logical name (the folder name without the trailing digest); a name may be
compiled several times with different parameters, so each run yields a multiset
of source signatures per name. A signature is the set of ``(relpath, sha256)``
pairs of one compiled program.

A name is compared only when both runs compiled the same number of programs for
it. The program count is a function of which tests ran, not of codegen, so equal
counts mean the same programs were generated in both runs and any difference in
their sources is codegen non-determinism. Names whose counts differ are normally
reported and skipped (a test failed or did not run in one of the runs). However,
if both runs are known to have completed cleanly (``runs_healthy=True``), a
differing count cannot be a failed test — it means the *number* of generated
programs is itself non-deterministic — so it is counted as a failure instead.
If no name is comparable the check raises rather than reporting a vacuous match.

Internally there are two parts: a comparison core (``check_determinism`` and its
helpers) that diffs two existing build caches, and an orchestration layer
(``run_determinism_check``) that first produces those caches by running a gt4py
pytest selection twice — shelling out to ``python -m pytest`` — and then compares
them. Both are exposed as CLI subcommands.

This is a standalone dev-scripts CLI: it is meant to be *executed*, not imported.
Run it via ``./scripts/run dace-determinism ...`` or directly through its uv
shebang; either way it runs inside the dev-scripts ``scripts`` environment, where
its dependency (``typer``) is available.

.. note::

   The ``ci-check`` subcommand runs the gt4py test suite, so it needs an
   interpreter that has gt4py installed. By default it uses the interpreter
   running this script — which, when launched via ``./scripts/run`` or the uv
   shebang, is the ``scripts`` environment, and that does **not** have gt4py.
   So ``ci-check`` is either driven by a gt4py-capable interpreter (this is what
   the ``test_next_dace_determinism`` nox session does: it executes the script
   with its session venv's python) or told one explicitly with ``--python``.
   The ``check`` subcommand has no such requirement — it only reads caches.

Usage::

    # Compare two existing build caches:
    ./scripts/run dace-determinism check --run1 PATH --run2 PATH \\
        [--diffs-dir DIR] [--report FILE] [--runs-healthy/--no-runs-healthy]

    # Run a pytest selection twice and compare (used by the
    # `test_next_dace_determinism` nox session, which omits --python because it
    # already executes the script with a gt4py-capable interpreter):
    ./scripts/run dace-determinism ci-check --python PATH/TO/gt4py/python \\
        --workdir DIR -- <pytest args>

Exit codes (both subcommands): 0 deterministic, 1 differs, 2 bad args /
unsupported backend / no source files / nothing comparable, 3 no programs
observed.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer


CODEGEN_DIR = "src"
SUPPORTED_BACKENDS = frozenset({"cpu", "cuda"})

SourceFileHash = tuple[str, str]  # (source file path, source file contents hash)
ProgramSignature = frozenset[SourceFileHash]  # all source files of one compiled program


class UnsupportedBackendError(RuntimeError):
    """A program's src/ contained a backend other than cpu/cuda."""


class NoProgramsObservedError(RuntimeError):
    """Neither cache contained any program folders."""


class NoSourceFilesObservedError(RuntimeError):
    """Programs were cached but none contained source files.

    Usually means DaCe's build folder mode is ``production`` (which strips
    src/ after compilation) instead of ``development``.
    """


class NoComparableProgramsError(RuntimeError):
    """Programs were cached but no logical name had a matching count in both runs.

    Nothing could be compared, so the result is inconclusive (e.g. one run failed
    wholesale, or the two runs share no name at an equal program count).
    """


class PytestRunError(RuntimeError):
    """A pytest run exited with an unexpected code (an infrastructure failure).

    Distinct from the comparison exceptions above so the CLI can map it to its
    own exit code without depending on ``except`` ordering against the shared
    ``RuntimeError`` base.
    """


class DeterminismError(RuntimeError):
    def __init__(self, message: str, results: list[ComparisonResult]) -> None:
        super().__init__(message)
        self.results = results


@dataclasses.dataclass
class ComparisonResult:
    """Comparison between the two runs of all programs sharing a logical name.

    Note: the same program can be compiled with different options (e.g.
    different static parameters). Different options and non-deterministic
    codegen both manifest in the same way — a different digest in the cache
    folder name — so a folder in one run cannot be paired with its counterpart
    in the other. Instead, this class collects all variants of one logical
    name into a single comparison result.
    """

    name: str
    comparable: bool  # both runs compiled the same number of programs
    match: bool  # comparable and the two signature multisets are identical
    only_in_run1: list[SourceFileHash]
    only_in_run2: list[SourceFileHash]
    count1: int
    count2: int

    @property
    def differs(self) -> bool:
        return self.comparable and not self.match

    @property
    def skipped(self) -> bool:
        return not self.comparable


def _is_failure(r: ComparisonResult, runs_healthy: bool | None) -> bool:
    """Whether a name counts as a determinism failure.

    A content difference always fails. A differing program count fails only when
    both runs completed cleanly: then it cannot be a missing test and so is a
    non-deterministic program count. Otherwise it is a tolerated skip.
    """
    return r.differs or (r.skipped and runs_healthy is True)


def _sha256_of_contents(path: Path) -> str:
    """Return the SHA-256 hexdigest of the file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan(cache_root: Path) -> tuple[dict[str, collections.Counter[ProgramSignature]], int]:
    """Return the signature multiset per logical name and the program count.

    Each program folder yields one ``ProgramSignature``: the set of
    ``(relative path, contents hash)`` pairs of every file under ``src/``. The
    contents hash is what the determinism check ultimately rests on — two
    compilations produced byte-identical code iff their signatures are equal,
    which ``compare`` tests via multiset equality.

    Program folders whose ``src/`` holds no source files are counted but
    excluded from the multisets, so they cannot manufacture mismatches.
    """
    if not cache_root.is_dir():
        return {}, 0

    by_name: dict[str, collections.Counter[ProgramSignature]] = collections.defaultdict(
        collections.Counter
    )
    n_folders = 0
    for folder in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        if not (folder / "program.sdfg").exists():
            continue
        n_folders += 1

        src_root = folder / CODEGEN_DIR
        sources: list[SourceFileHash] = []
        if src_root.is_dir():
            for backend in sorted(d for d in src_root.iterdir() if d.is_dir()):
                if backend.name not in SUPPORTED_BACKENDS:
                    raise UnsupportedBackendError(
                        f"unsupported dace backend `{backend.name}/` under {src_root}; "
                        f"supported top-level backends are {sorted(SUPPORTED_BACKENDS)} "
                        f"(HIP lives under cuda/hip/)"
                    )
            for path in sorted(src_root.rglob("*")):
                if path.is_file():
                    sources.append((path.relative_to(folder).as_posix(), _sha256_of_contents(path)))

        if sources:
            by_name[m.group("name")][frozenset(sources)] += 1
    return dict(by_name), n_folders


def _diagnose_empty(cache_root: Path) -> str:
    if not cache_root.exists():
        return "path does not exist"
    if not cache_root.is_dir():
        return "path is not a directory"
    subdirs = [p for p in cache_root.iterdir() if p.is_dir()]
    if not subdirs:
        return "no subdirectories (nothing cached)"
    if not any((p / "program.sdfg").exists() for p in subdirs):
        return "no SDFG build directories found"
    return "program folders present but none could be read"


def compare(
    run1: dict[str, collections.Counter[ProgramSignature]],
    run2: dict[str, collections.Counter[ProgramSignature]],
) -> list[ComparisonResult]:
    results = []
    for name in sorted(set(run1) | set(run2)):
        bag1 = run1.get(name, collections.Counter())
        bag2 = run2.get(name, collections.Counter())
        count1 = sum(bag1.values())
        count2 = sum(bag2.values())

        comparable = count1 == count2
        match = comparable and bag1 == bag2

        extra1 = bag1 - bag2
        extra2 = bag2 - bag1
        results.append(
            ComparisonResult(
                name=name,
                comparable=comparable,
                match=match,
                only_in_run1=sorted(pair for sig in extra1.elements() for pair in sig),
                only_in_run2=sorted(pair for sig in extra2.elements() for pair in sig),
                count1=count1,
                count2=count2,
            )
        )
    return results


def write_diffs(results: list[ComparisonResult], diffs_dir: Path) -> None:
    for r in results:
        if r.match:
            continue
        c1 = collections.Counter(r.only_in_run1)
        c2 = collections.Counter(r.only_in_run2)
        relpaths = sorted({rel for rel, _ in (c1 - c2)} | {rel for rel, _ in (c2 - c1)})
        diffs_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", r.name)[:200]
        (diffs_dir / f"{safe}.txt").write_text("\n".join([r.name, *relpaths]) + "\n")


def render_report(results: list[ComparisonResult], *, runs_healthy: bool | None = None) -> str:
    n_total = len(results)
    n_match = sum(1 for r in results if r.match)
    n_differ = sum(1 for r in results if r.differs)
    n_skipped = sum(1 for r in results if r.skipped)
    n_comparable = n_total - n_skipped
    n_failed = sum(1 for r in results if _is_failure(r, runs_healthy))
    n_count = n_failed - n_differ  # count differences treated as failures
    skips_fail = runs_healthy is True

    skip_label = "count-mismatch" if skips_fail else "skipped"
    lines = [
        f"Programs: {n_total}    comparable: {n_comparable}    matches: {n_match}    "
        f"differs: {n_differ}    {skip_label}: {n_skipped}",
        "",
    ]
    for r in results:
        if r.match:
            tag = "MATCH "
        elif r.differs:
            tag = "DIFFER"
        elif _is_failure(r, runs_healthy):
            tag = "COUNT "
        else:
            tag = "SKIP  "
        lines.append(f"  [{tag}] {r.name}")

    if n_skipped:
        if skips_fail:
            note = (
                f"{n_skipped} program(s) produced a different number of generated "
                "variants between two clean runs — the program count itself is "
                "non-deterministic (counted as a failure)."
            )
        else:
            note = (
                f"{n_skipped} program(s) skipped: the two runs compiled a different "
                "number of them, so they are not comparable. Re-run until both runs "
                "are clean to cover them."
            )
        lines += ["", note]

    lines.append("")
    if n_total == 0:
        lines.append("RESULT: no programs observed.")
    elif n_failed > 0:
        bits = []
        if n_differ:
            bits.append(f"{n_differ}/{n_comparable} comparable program(s) differ")
        if n_count:
            bits.append(f"{n_count} program(s) with non-deterministic count")
        lines.append("RESULT: NON-DETERMINISTIC CODEGEN — " + "; ".join(bits) + ".")
    elif n_comparable == 0:
        lines.append(
            f"RESULT: NOTHING COMPARABLE — all {n_skipped} program(s) had differing "
            f"counts between runs; nothing was verified. Re-run with both runs clean."
        )
    elif n_skipped == 0:
        lines.append(f"RESULT: codegen deterministic — {n_match} program(s) match.")
    else:
        lines.append(
            f"RESULT: codegen deterministic across {n_match} comparable program(s); "
            f"{n_skipped} skipped (re-run for full coverage)."
        )
    return "\n".join(lines) + "\n"


def check_determinism(
    cache1: Path,
    cache2: Path,
    *,
    runs_healthy: bool | None = None,
    diffs_dir: Path | None = None,
    report_path: Path | None = None,
) -> list[ComparisonResult]:
    bags1, n_folders1 = _scan(cache1)
    bags2, n_folders2 = _scan(cache2)
    results = compare(bags1, bags2)

    if diffs_dir is not None:
        write_diffs(results, diffs_dir)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(results, runs_healthy=runs_healthy))

    if n_folders1 == 0 and n_folders2 == 0:
        raise NoProgramsObservedError(
            "no programs observed in either cache:\n"
            f"  run1 ({cache1}): {_diagnose_empty(cache1)}\n"
            f"  run2 ({cache2}): {_diagnose_empty(cache2)}"
        )
    if not results:
        raise NoSourceFilesObservedError(
            f"program(s) cached ({n_folders1} in run1, {n_folders2} in run2) but none "
            f"contain source files under {CODEGEN_DIR}/. Set "
            f"DACE_compiler_build_folder_mode=development so the generated code is kept "
            f"after compilation."
        )

    n_differ = sum(1 for r in results if r.differs)
    n_skipped = sum(1 for r in results if r.skipped)
    n_comparable = len(results) - n_skipped
    n_failed = sum(1 for r in results if _is_failure(r, runs_healthy))

    # Inconclusive only when nothing was comparable and the count mismatches
    # cannot be blamed on codegen; in a clean pair those mismatches are failures.
    if n_comparable == 0 and n_failed == 0:
        raise NoComparableProgramsError(
            f"no logical name had a matching program count in both runs "
            f"({n_skipped} name(s) skipped), so nothing could be compared. "
            f"Likely one run failed wholesale; re-run with both runs clean."
        )

    if n_failed > 0:
        bits = []
        if n_differ:
            bits.append(f"{n_differ}/{n_comparable} comparable program(s) differ")
        n_count = n_failed - n_differ
        if n_count:
            bits.append(
                f"{n_count} program(s) produced a different number of generated "
                f"variants between two clean runs"
            )
        raise DeterminismError("DaCe codegen is non-deterministic: " + "; ".join(bits), results)
    return results


# -- Orchestration --
#
# The `run_determinism_check` function below runs a gt4py pytest selection twice
# with an isolated build cache per run, then compares the generated sources. It
# uses only the stdlib + subprocess (it shells out to `python -m pytest`), so it
# carries no nox dependency and runs wherever gt4py's test suite can run.

CACHE_SUBDIR = ".gt4py_cache"
PYTEST_TOLERATED_EXIT_CODES = frozenset({0, 1, 5})  # ok, tests failed, no tests collected


def _run_is_healthy(junit_xml: Path) -> bool:
    """Whether a pytest run collected tests and none failed or errored.

    Read from the JUnit XML (written regardless of outcome) so the comparator can
    tell a legitimate skip (a test failed -> differing program count) from a
    genuine non-deterministic program count (counts differ in a clean pair).
    """
    import xml.etree.ElementTree as ET

    if not junit_xml.is_file():
        return False
    root = ET.parse(junit_xml).getroot()
    tests = failures = errors = 0
    for suite in root.iter("testsuite"):
        tests += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
    return tests > 0 and failures == 0 and errors == 0


def _env_for_run(run_dir: Path) -> dict[str, str]:
    # gt4py.next appends `.gt4py_cache` to GT4PY_BUILD_CACHE_DIR, so we pass the
    # parent directory and the cache lands at .gt4py_cache/ underneath. Setting
    # GT4PY_BUILD_CACHE_LIFETIME to `persistent` keeps the cache around long
    # enough for the comparison to read it.
    #
    # Setting DACE_compiler_build_folder_mode to `development` is REQUIRED: gt4py
    # configures dace to `production` mode by default, which cleans up the dace
    # build folder after compilation — leaving only the compiled .so and
    # stripping the codegen sources we need to diff. Forcing `development` keeps
    # `src/...` around so the checker has codegen to compare. (See src/gt4py/next/
    # program_processors/runners/dace/workflow/common.py for the upstream config
    # this overrides; the comment there documents this env var as the escape hatch.)
    return {
        "GT4PY_BUILD_CACHE_DIR": str(run_dir),
        "GT4PY_BUILD_CACHE_LIFETIME": "persistent",
        "DACE_compiler_build_folder_mode": "development",
    }


def run_determinism_check(
    pytest_args: Sequence[str],
    *,
    workdir: Path,
    python: str = sys.executable,
    dacecache: Path | None = None,
    self_check: bool = True,
) -> list[ComparisonResult]:
    """Run ``pytest`` twice with an isolated cache per run, then compare codegen.

    ``pytest_args`` are the arguments passed to ``python -m pytest`` (without the
    leading ``pytest``) for each of the two runs. ``python`` is the interpreter
    used to run the suite (defaults to the current one, i.e. the environment that
    has gt4py installed when this script is executed there). Each run sets
    ``GT4PY_BUILD_CACHE_DIR`` so its cache lands at ``<run_dir>/.gt4py_cache/``.

    ``dacecache`` is DaCe's build-cache directory, wiped before each run so the
    two runs cannot share build artifacts. It defaults to ``<cwd>/.dacecache`` —
    where DaCe writes it relative to the test process's working directory — and
    must be overridden if a custom DaCe build folder is configured.

    The two per-run caches and ``dacecache`` are removed afterwards; ``diffs/``
    and ``report.txt`` under ``workdir`` are kept as debugging artifacts. Returns
    the comparison results, or raises the same exceptions as ``check_determinism``
    (plus ``PytestRunError`` if a pytest run fails for an infrastructure reason).
    """
    workdir = workdir.expanduser().resolve()
    dacecache = (dacecache or Path.cwd() / ".dacecache").expanduser().resolve()

    # Self-check the comparator first: a broken script aborts here, before the
    # two expensive test-suite runs.
    if self_check:
        test_file = (
            Path(__file__).resolve().parent.parent / "tests" / "python" / "test_dace_determinism.py"
        )
        subprocess.run([python, "-m", "pytest", "-q", str(test_file)], check=True)

    if workdir.exists():
        shutil.rmtree(workdir)
    run1_dir = workdir / "run1"
    run2_dir = workdir / "run2"
    run1_dir.mkdir(parents=True)
    run2_dir.mkdir(parents=True)

    def wipe_dacecache() -> None:
        if dacecache.exists():
            shutil.rmtree(dacecache, ignore_errors=True)

    runs_healthy = True
    try:
        for run_dir in (run1_dir, run2_dir):
            wipe_dacecache()
            junit_xml = run_dir / "pytest-junit.xml"
            result = subprocess.run(
                [python, "-m", "pytest", *pytest_args, f"--junit-xml={junit_xml}"],
                env={**os.environ, **_env_for_run(run_dir)},
            )
            # Tolerate test failures / empty collection: a failed test only
            # changes how a count mismatch is classified, never whether content
            # differences are reported. Any other code is an infrastructure error.
            if result.returncode not in PYTEST_TOLERATED_EXIT_CODES:
                raise PytestRunError(
                    f"pytest exited with unexpected code {result.returncode} for run "
                    f"{run_dir.name}; cannot assess determinism."
                )
            runs_healthy = _run_is_healthy(junit_xml) and runs_healthy

        return check_determinism(
            run1_dir / CACHE_SUBDIR,
            run2_dir / CACHE_SUBDIR,
            runs_healthy=runs_healthy,
            diffs_dir=workdir / "diffs",
            report_path=workdir / "report.txt",
        )
    finally:
        # Reclaim disk after the comparison. The two per-run caches are ~hundreds
        # of MB each in development mode. We always keep `workdir/diffs/` and
        # `workdir/report.txt` — those are the artifacts a maintainer needs to
        # debug a failure; the raw caches are reproducible by rerunning.
        for tbd in (run1_dir, run2_dir, dacecache):
            shutil.rmtree(tbd, ignore_errors=True)


# Exit-code mapping shared by the `check` and `run` CLI subcommands. (0 is the
# success code, returned implicitly; the values below mirror the docstring.)
_EXIT_CODES: dict[type[Exception], int] = {
    DeterminismError: 1,
    UnsupportedBackendError: 2,
    NoSourceFilesObservedError: 2,
    NoComparableProgramsError: 2,
    NoProgramsObservedError: 3,
}


# -- CLI --
#
# This module always runs inside the dev-scripts ``scripts`` environment (via
# ``./scripts/run`` or its uv shebang), so ``typer`` is always importable.

cli = typer.Typer(no_args_is_help=True, name="dace-determinism", help=__doc__)


@cli.command()
def check(
    run1: Annotated[Path, typer.Option("--run1", metavar="PATH", help="First cache root.")],
    run2: Annotated[Path, typer.Option("--run2", metavar="PATH", help="Second cache root.")],
    diffs_dir: Annotated[
        Path | None,
        typer.Option(
            "--diffs-dir",
            metavar="PATH",
            help="If set, write per-program mismatch reports here.",
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", metavar="PATH", help="If set, write the summary report here."),
    ] = None,
    runs_healthy: Annotated[
        bool | None,
        typer.Option(
            "--runs-healthy/--no-runs-healthy",
            help="Whether both runs completed cleanly. When set, a differing program "
            "count is treated as a failure (non-deterministic count) instead of a skip.",
        ),
    ] = None,
) -> None:
    """Compare two gt4py.next build caches for deterministic DaCe codegen."""
    try:
        results = check_determinism(
            run1.expanduser().resolve(),
            run2.expanduser().resolve(),
            runs_healthy=runs_healthy,
            diffs_dir=diffs_dir.expanduser().resolve() if diffs_dir else None,
            report_path=report.expanduser().resolve() if report else None,
        )
    except DeterminismError as e:
        if report is None:
            typer.echo(render_report(e.results, runs_healthy=runs_healthy))
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(_EXIT_CODES[type(e)]) from e
    except (
        UnsupportedBackendError,
        NoProgramsObservedError,
        NoSourceFilesObservedError,
        NoComparableProgramsError,
    ) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(_EXIT_CODES[type(e)]) from e
    if report is None:
        typer.echo(render_report(results, runs_healthy=runs_healthy))


@cli.command("ci-check")
def ci_check(
    pytest_args: Annotated[
        list[str] | None,
        typer.Argument(
            help="Arguments passed to `python -m pytest` for each of the two runs "
            "(pass them after `--`)."
        ),
    ] = None,
    workdir: Annotated[
        Path,
        typer.Option(
            "--workdir",
            metavar="PATH",
            help="Working directory for the per-run caches, diffs/, and report.txt.",
        ),
    ] = Path("_dace_deterministic_codegen"),
    python: Annotated[
        str,
        typer.Option(
            "--python",
            metavar="EXE",
            help="Interpreter used to run the gt4py test suite; it must have gt4py "
            "installed. Defaults to the interpreter running this script (which only "
            "has gt4py when the script is launched by a gt4py-capable interpreter, "
            "e.g. the nox session venv — not the bare `./scripts/run` environment).",
        ),
    ] = sys.executable,
    dacecache: Annotated[
        Path | None,
        typer.Option(
            "--dacecache",
            metavar="PATH",
            help="DaCe's build-cache directory, wiped between runs so they don't "
            "share artifacts. Defaults to ./.dacecache (where DaCe writes it); "
            "override only if a custom DaCe build folder is configured.",
        ),
    ] = None,
    self_check: Annotated[
        bool,
        typer.Option(
            "--self-check/--no-self-check",
            help="Run the comparator's own unit tests before the expensive runs.",
        ),
    ] = True,
) -> None:
    """Run a pytest selection twice and verify DaCe codegen is byte-identical."""
    workdir_abs = workdir.expanduser().resolve()
    try:
        run_determinism_check(
            pytest_args or [],
            workdir=workdir,
            python=python,
            dacecache=dacecache,
            self_check=self_check,
        )
    except subprocess.CalledProcessError as e:
        typer.echo(f"error: comparator self-check failed (exit {e.returncode}).", err=True)
        raise typer.Exit(2) from e
    except PytestRunError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    except (
        DeterminismError,
        UnsupportedBackendError,
        NoProgramsObservedError,
        NoSourceFilesObservedError,
        NoComparableProgramsError,
    ) as e:
        typer.echo(
            f"error: {e}\nSee {workdir_abs / 'report.txt'} and {workdir_abs / 'diffs'}/",
            err=True,
        )
        raise typer.Exit(_EXIT_CODES[type(e)]) from e
    typer.echo((workdir_abs / "report.txt").read_text())


if __name__ == "__main__":
    cli()
