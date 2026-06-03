#!/usr/bin/env python3
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
their sources is codegen non-determinism. Names whose counts differ (a test
failed or did not run in one of the runs) are reported and skipped. If no name
is comparable the check raises rather than reporting a vacuous match.

Usage::

    from scripts.dace_deterministic_codegen import check_determinism
    check_determinism(run1_cache, run2_cache, diffs_dir=..., report_path=...)

CLI::

    python scripts/dace_deterministic_codegen.py --run1 PATH --run2 PATH \\
        [--diffs-dir DIR] [--report FILE]

Exit codes: 0 deterministic, 1 differs, 2 bad args / unsupported backend /
no source files / nothing comparable, 3 no programs observed.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import re
import sys
from pathlib import Path


PROGRAM_FOLDER_RE = re.compile(r"^(?P<name>.+)_[0-9a-f]{64}$")
CODEGEN_DIR = "src"
SUPPORTED_BACKENDS = frozenset({"cpu", "cuda"})


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
    def __init__(self, message: str, results: list[NameResult]) -> None:
        super().__init__(message)
        self.results = results


@dataclasses.dataclass
class NameResult:
    name: str
    comparable: bool  # both runs compiled the same number of programs
    match: bool  # comparable and the two signature multisets are identical
    only_in_run1: list[tuple[str, str]]
    only_in_run2: list[tuple[str, str]]
    count1: int
    count2: int

    @property
    def differs(self) -> bool:
        return self.comparable and not self.match

    @property
    def skipped(self) -> bool:
        return not self.comparable


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan(cache_root: Path) -> tuple[dict[str, collections.Counter], int]:
    """Return the signature multiset per logical name and the program count.

    Program folders whose ``src/`` holds no source files are counted but
    excluded from the multisets, so they cannot manufacture mismatches.
    """
    if not cache_root.is_dir():
        return {}, 0

    by_name: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_folders = 0
    for folder in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        m = PROGRAM_FOLDER_RE.match(folder.name)
        if not m:
            continue
        n_folders += 1

        src_root = folder / CODEGEN_DIR
        sources: list[tuple[str, str]] = []
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
                    sources.append((path.relative_to(folder).as_posix(), _sha256(path)))

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
    run1: dict[str, collections.Counter],
    run2: dict[str, collections.Counter],
) -> list[NameResult]:
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
            NameResult(
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


def write_diffs(results: list[NameResult], diffs_dir: Path) -> None:
    for r in results:
        if r.match:
            continue
        kind = (
            "generated sources differ"
            if r.differs
            else f"skipped: program counts differ between runs ({r.count1} vs {r.count2})"
        )
        lines = [
            f"name: {r.name}",
            f"classification: {kind}",
            f"run1 produced {r.count1} program(s), run2 produced {r.count2}",
            "",
            "generated sources in run1 with no match in run2:",
        ]
        lines += [f"  {rel}  {sha}" for rel, sha in r.only_in_run1] or ["  (none)"]
        lines += ["", "generated sources in run2 with no match in run1:"]
        lines += [f"  {rel}  {sha}" for rel, sha in r.only_in_run2] or ["  (none)"]
        diffs_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", r.name)[:200]
        (diffs_dir / f"{safe}.txt").write_text("\n".join(lines) + "\n")


def render_report(results: list[NameResult]) -> str:
    n_total = len(results)
    n_differ = sum(1 for r in results if r.differs)
    n_skipped = sum(1 for r in results if r.skipped)
    n_match = n_total - n_differ - n_skipped
    n_comparable = n_total - n_skipped

    lines = [
        f"Programs: {n_total}    comparable: {n_comparable}    matches: {n_match}    "
        f"differs: {n_differ}    skipped: {n_skipped}",
        "",
    ]
    for r in results:
        tag = "MATCH " if r.match else "DIFFER" if r.differs else "SKIP  "
        lines.append(f"  [{tag}] {r.name}  (run1: {r.count1}, run2: {r.count2})")
        if r.differs:
            lines += [f"           only in run1: {rel}" for rel, _ in r.only_in_run1]
            lines += [f"           only in run2: {rel}" for rel, _ in r.only_in_run2]

    if n_skipped:
        lines += [
            "",
            f"{n_skipped} program(s) skipped: the two runs compiled a different number "
            "of them, so they are not comparable. Re-run until both runs are clean to "
            "cover them.",
        ]

    lines.append("")
    if n_total == 0:
        lines.append("RESULT: no programs observed.")
    elif n_differ > 0:
        suffix = f" ({n_skipped} skipped)" if n_skipped else ""
        lines.append(
            f"RESULT: NON-DETERMINISTIC CODEGEN — {n_differ}/{n_comparable} "
            f"comparable program(s) differ{suffix}."
        )
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
    diffs_dir: Path | None = None,
    report_path: Path | None = None,
) -> list[NameResult]:
    bags1, n_folders1 = _scan(cache1)
    bags2, n_folders2 = _scan(cache2)
    results = compare(bags1, bags2)

    if diffs_dir is not None:
        write_diffs(results, diffs_dir)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(results))

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

    if n_comparable == 0:
        raise NoComparableProgramsError(
            f"no logical name had a matching program count in both runs "
            f"({n_skipped} name(s) skipped), so nothing could be compared. "
            f"Likely one run failed wholesale; re-run with both runs clean."
        )
    if n_differ > 0:
        extra = f" ({n_skipped} skipped)" if n_skipped else ""
        raise DeterminismError(
            f"DaCe codegen is non-deterministic: {n_differ}/{n_comparable} "
            f"comparable program(s) differ{extra}",
            results,
        )
    return results


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dace_deterministic_codegen",
        description="Compare two gt4py.next build caches for deterministic DaCe codegen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run1", required=True, type=Path, metavar="PATH", help="First cache root.")
    p.add_argument("--run2", required=True, type=Path, metavar="PATH", help="Second cache root.")
    p.add_argument(
        "--diffs-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, write per-program mismatch reports here.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, write the summary report here.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
    except (NoSourceFilesObservedError, NoComparableProgramsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except DeterminismError as e:
        if args.report is None:
            print(render_report(e.results))
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.report is None:
        print(render_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
