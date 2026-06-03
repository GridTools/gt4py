#!/usr/bin/env python3
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that gt4py's DaCe backend generates identical source across two runs.

gt4py.next caches each compiled program at ``<root>/<name>_<sha256>/src/...``.
The ``<sha256>`` is derived from the generated source, so non-deterministic
codegen yields a different folder name each run. Programs are therefore
compared by logical name (the folder name without the trailing digest), and a
program may be compiled several times under one name with different baked
parameters.

For each logical name the check collects the multiset of generated-source
signatures produced in each run and compares them. A signature is the set of
``(relative-path, sha256)`` pairs of one compiled program. Two outcomes are
distinguished when the bags are not equal:

* **Divergence** — each run produced a signature the other never produced
  (novel content on *both* sides). This is the fingerprint of nondeterministic
  codegen: a computation rendered one way in run1 and a different way in run2.
  Always a failure.
* **One-sided gap** — extra signatures or extra copies appear on only one
  side, with no unmatched content on the other. This is what a run compiling a
  different *set* of programs looks like (e.g. a test erroring before codegen
  in one run only, under ``-n auto``). It is reported but, like a whole program
  missing from one run, tolerated unless ``tolerate_missing=False``.

Programs whose ``src/`` holds no source files are excluded from the bags (they
carry no codegen to compare); if no program anywhere has sources the check
raises rather than reporting a vacuous match.

Usage::

    from scripts.dace_deterministic_codegen import check_determinism
    check_determinism(run1_cache, run2_cache,
                      diffs_dir=..., report_path=...)

CLI::

    python scripts/dace_deterministic_codegen.py --run1 PATH --run2 PATH \\
        [--diffs-dir DIR] [--report FILE] [--no-tolerate-missing]

Exit codes: 0 deterministic, 1 differs, 2 bad args / unsupported backend /
no source files, 3 no programs observed.
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


class DeterminismError(RuntimeError):
    def __init__(self, message: str, results: list[NameResult]) -> None:
        super().__init__(message)
        self.results = results


@dataclasses.dataclass
class NameResult:
    name: str
    match: bool
    divergent: bool
    missing_on_one_side: bool
    only_in_run1: list[tuple[str, str]]
    only_in_run2: list[tuple[str, str]]
    count1: int
    count2: int

    @property
    def tolerable(self) -> bool:
        """A non-match that is not divergence: extra programs on one side only."""
        return not self.match and not self.divergent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan(cache_root: Path) -> tuple[dict[str, collections.Counter], int]:
    """Return ``(bags, n_program_folders)`` for one cache.

    ``bags`` maps each logical name to a Counter of program signatures, where a
    signature is the frozenset of ``(relpath, sha256)`` source pairs. Program
    folders whose ``src/`` yields no source files are counted in
    ``n_program_folders`` but excluded from ``bags`` — they carry no codegen, so
    folding them in as an empty signature would manufacture spurious mismatches.
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
        name = m.group("name")

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
                    rel = path.relative_to(folder).as_posix()
                    sources.append((rel, _sha256(path)))

        if sources:
            by_name[name][frozenset(sources)] += 1
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

        extra1 = bag1 - bag2
        extra2 = bag2 - bag1
        match = not extra1 and not extra2

        # Novel = a signature wholly absent from the other run (not merely
        # produced a different number of times). Divergence — the signature of
        # nondeterministic codegen — is novel content appearing on *both* sides:
        # run1 rendered something run2 never did and vice versa. Novelty on only
        # one side is just a run compiling more programs than the other.
        novel1 = any(sig not in bag2 for sig in extra1)
        novel2 = any(sig not in bag1 for sig in extra2)
        divergent = novel1 and novel2
        missing = not match and (count1 == 0 or count2 == 0)

        only1 = sorted(rel_sha for sig in extra1.elements() for rel_sha in sig)
        only2 = sorted(rel_sha for sig in extra2.elements() for rel_sha in sig)

        results.append(
            NameResult(
                name=name,
                match=match,
                divergent=divergent,
                missing_on_one_side=missing,
                only_in_run1=only1,
                only_in_run2=only2,
                count1=count1,
                count2=count2,
            )
        )
    return results


def write_diffs(results: list[NameResult], diffs_dir: Path) -> None:
    for r in results:
        if r.match:
            continue
        kind = "divergent (nondeterministic)" if r.divergent else "one-sided (tolerable)"
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


def render_report(results: list[NameResult], *, tolerate_missing: bool = False) -> str:
    n_total = len(results)
    n_divergent = sum(1 for r in results if r.divergent)
    n_missing = sum(1 for r in results if r.missing_on_one_side)
    n_gap = sum(1 for r in results if r.tolerable and not r.missing_on_one_side)
    n_match = n_total - n_divergent - n_missing - n_gap
    n_tolerable = n_missing + n_gap

    lines = [
        f"Programs: {n_total}    matches: {n_match}    differs: {n_divergent}    "
        f"only-in-one-run: {n_missing}    count-gap: {n_gap}",
        "",
    ]
    for r in results:
        if r.match:
            tag = "MATCH "
        elif r.divergent:
            tag = "DIFFER"
        elif r.missing_on_one_side:
            tag = "ONE-OF"
        else:
            tag = "GAP   "
        lines.append(f"  [{tag}] {r.name}  (run1: {r.count1}, run2: {r.count2})")
        if r.divergent:
            lines += [f"           only in run1: {rel}" for rel, _ in r.only_in_run1]
            lines += [f"           only in run2: {rel}" for rel, _ in r.only_in_run2]

    lines.append("")
    if n_total == 0:
        lines.append("RESULT: no programs observed.")
    elif n_divergent == 0 and n_tolerable == 0:
        lines.append(f"RESULT: codegen deterministic — {n_match} program(s) match.")
    elif n_divergent == 0 and tolerate_missing:
        lines.append(
            f"RESULT: codegen deterministic across {n_match} shared program(s); "
            f"{n_tolerable} compiled in only one run (tolerated)."
        )
    elif n_divergent == 0:
        lines.append(
            f"RESULT: FAILED (strict) — {n_tolerable} program(s) compiled in only one run."
        )
    else:
        suffix = f" (plus {n_tolerable} compiled in only one run)" if n_tolerable else ""
        lines.append(
            f"RESULT: NON-DETERMINISTIC CODEGEN — {n_divergent}/{n_total} program(s) differ{suffix}."
        )
    return "\n".join(lines) + "\n"


def check_determinism(
    cache1: Path,
    cache2: Path,
    *,
    diffs_dir: Path | None = None,
    report_path: Path | None = None,
    tolerate_missing: bool = True,
) -> list[NameResult]:
    bags1, n_folders1 = _scan(cache1)
    bags2, n_folders2 = _scan(cache2)
    results = compare(bags1, bags2)

    if diffs_dir is not None:
        write_diffs(results, diffs_dir)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(results, tolerate_missing=tolerate_missing))

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

    n_divergent = sum(1 for r in results if r.divergent)
    n_tolerable = sum(1 for r in results if r.tolerable)
    n_failed = n_divergent if tolerate_missing else n_divergent + n_tolerable

    if n_failed > 0:
        if tolerate_missing:
            msg = (
                f"DaCe codegen is non-deterministic: {n_divergent}/{len(results)} "
                f"program(s) differ (plus {n_tolerable} compiled in only one run, ignored)"
            )
        elif n_divergent == 0:
            msg = (
                f"determinism check failed (strict): {n_tolerable}/{len(results)} "
                f"program(s) compiled in only one run"
            )
        else:
            msg = (
                f"DaCe codegen is non-deterministic: {n_divergent}/{len(results)} differ "
                f"(plus {n_tolerable} compiled in only one run)"
            )
        raise DeterminismError(msg, results)
    return results


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dace_deterministic_codegen",
        description="Compare two gt4py.next build caches for deterministic DaCe codegen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run1",
        required=True,
        type=Path,
        metavar="PATH",
        help="First cache root (the .gt4py_cache/ directory).",
    )
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
    p.add_argument(
        "--tolerate-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore programs compiled in only one run.",
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
        if args.report is None:
            print(render_report(e.results, tolerate_missing=args.tolerate_missing))
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.report is None:
        print(render_report(results, tolerate_missing=args.tolerate_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
