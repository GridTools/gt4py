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

The library API below (``check_determinism`` and friends) is intentionally
stdlib-only so it can be imported from ``noxfile.py`` and the nox session venvs,
neither of which has the dev-scripts dependencies installed.

Usage::

    from dace_deterministic_codegen import check_determinism
    check_determinism(run1_cache, run2_cache, runs_healthy=True,
                      diffs_dir=..., report_path=...)

CLI::

    ./scripts/run dace-determinism check --run1 PATH --run2 PATH \\
        [--diffs-dir DIR] [--report FILE] [--runs-healthy/--no-runs-healthy]

Exit codes: 0 deterministic, 1 differs, 2 bad args / unsupported backend /
no source files / nothing comparable, 3 no programs observed.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import re
from pathlib import Path


PROGRAM_FOLDER_RE = re.compile(r"^(?P<name>.+)_[0-9a-f]{64}$")
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
        m = PROGRAM_FOLDER_RE.match(folder.name)
        if not m:
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
    if not any(PROGRAM_FOLDER_RE.match(p.name) for p in subdirs):
        return "no subdirectory matches `<name>_<64-hex-digest>/`"
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


# -- CLI --
#
# ``typer`` lives in the dev-scripts ``scripts`` dependency group, which is only
# present in the uv environment created by this file's shebang (and used by
# ``scripts/run``). The library API above must stay importable from stdlib-only
# interpreters (``noxfile.py`` and the nox session venvs import it directly), so
# the CLI is guarded: those importers never touch ``cli``, and ``scripts/run``
# only ever imports this module from inside the uv scripts env.
try:
    import typer
except ImportError:
    typer = None  # type: ignore[assignment]

if typer is not None:
    from typing import Annotated

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
        except UnsupportedBackendError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(2) from e
        except NoProgramsObservedError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(3) from e
        except (NoSourceFilesObservedError, NoComparableProgramsError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(2) from e
        except DeterminismError as e:
            if report is None:
                typer.echo(render_report(e.results, runs_healthy=runs_healthy))
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
        if report is None:
            typer.echo(render_report(results, runs_healthy=runs_healthy))


if __name__ == "__main__":
    if typer is None:
        raise SystemExit(
            "error: this CLI requires `typer`; run it via `./scripts/run` or the uv shebang."
        )
    cli()
