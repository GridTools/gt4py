# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for inspecting and pruning the gt4py.next caches for generated code.

gt4py.next keeps two kinds of persistent cache side by side under the cache base
directory (see `gt4py.next.otf.compilation.cache`):

- **Build caches**, one folder per compiled program variant, holding the build
  artifacts and the compiled library.
- **Translation caches**, one pickled file per translated program, holding the
  output of the translation step: the optimized SDFG for the DaCe backend, the
  generated source for the gtfn backend.

The translation cache is keyed by a fingerprint of the program plus the gt4py
version, not by the gt4py sources. Editing a transformation or an optimization
pass without changing either therefore leaves the fingerprint intact: the cached
translation is replayed and the pass never runs, while the build step still
recompiles and refreshes the library's mtime. A fresh library is thus evidence of
recompilation, never of re-translation. The two caches hit and miss independently,
so the `status` command reports a verdict for each.

Run it in the environment that produced the cache, from the working directory the
cached run used (the default cache base is `<cwd>/.gt4py_cache`), either through
the installed `gt4py-next-cache` command or as a module::

    gt4py-next-cache status --program 'apply_diffusion_*'
    python -m gt4py.next.gt_cache_manager delete --program 'apply_diffusion_*' --yes
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import fnmatch
import json
import pathlib
import pickle
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from gt4py._core import locking
from gt4py.next import config
from gt4py.next.otf import stages
from gt4py.next.otf.compilation import build_data, cache
from gt4py.next.otf.compilation.build_systems import compiledb


ENTRY_SUFFIX: Final[str] = ".pkl"

EXIT_OK: Final[int] = 0
#: Nothing was done: a selector matched nothing, the user declined, or a gate tripped.
EXIT_NOTHING_DONE: Final[int] = 1
#: The command could not run: bad arguments, or a confirmation that cannot be obtained.
EXIT_ERROR: Final[int] = 2

_BUILD_FOLDER_RE: Final[re.Pattern[str]] = re.compile(cache.CACHE_FOLDER_NAME_PATTERN)


class CacheDirError(RuntimeError):
    """The given path is not usable as a gt4py.next cache base directory."""


@dataclasses.dataclass(frozen=True)
class Entry:
    """One translation cache entry, with what could be recovered from it.

    `program` is `None` exactly when the payload could not be unpickled, in which
    case `error` describes why. Such an entry is dead weight rather than a
    replay risk: the runtime cannot read it either and recomputes.
    """

    backend: str
    key: str
    path: pathlib.Path
    size: int
    mtime: float
    program: str | None
    error: str | None


@dataclasses.dataclass(frozen=True)
class BuildDir:
    """One build cache folder, with what its name and contents say about it."""

    program: str
    path: pathlib.Path
    version_id: str
    complete: bool

    @property
    def usable(self) -> bool:
        """Whether a run in this environment could still be served by this folder.

        A folder built under a different build-cache version can never be hit
        again: the version salts the folder name, so the next run looks for a
        different folder and builds it.
        """
        return self.complete and self.version_id == config.BUILD_CACHE_VERSION_ID


@dataclasses.dataclass(frozen=True)
class ProgramStatus:
    """What the next run of one program can do with each of the two caches.

    Translation and build are cached separately and hit or miss independently, so
    they are reported as two verdicts rather than one. Both are counted per
    program name, which is all the caches record about their contents, while a
    hit is decided by a fingerprint taken at run time. `replays` and `recompiles`
    are therefore not symmetric: `not replays` and `recompiles` are guarantees,
    since nothing on disk can serve this program, while the opposite only says
    that something on disk could serve it.
    """

    program: str
    entries: Mapping[str, int]
    build_dirs: int
    usable_build_dirs: int

    @property
    def replays(self) -> bool:
        """Whether a cached translation is available to replay."""
        return sum(self.entries.values()) > 0

    @property
    def recompiles(self) -> bool:
        """Whether no usable build is available, so the library is built again."""
        return self.usable_build_dirs == 0


def get_cache_base(cache_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Return the cache base directory, or `cache_dir` if one is given.

    Args:
        cache_dir: Explicit cache base. It is validated to look like a gt4py.next
            cache, since the delete commands remove files underneath it.

    Returns:
        The resolved cache base directory. It does not necessarily exist: a
        session-lifetime cache base is created lazily.

    Raises:
        CacheDirError: If `cache_dir` does not exist or does not look like a
            gt4py.next cache base.
    """
    if cache_dir is None:
        return cache.get_cache_base_path(config.BUILD_CACHE_LIFETIME)

    path = cache_dir.expanduser().resolve()
    if not path.is_dir():
        raise CacheDirError(f"'{path}' is not a directory.")
    if not (
        path.name == config.BUILD_CACHE_DIR.name
        or any((path / name).is_dir() for name in cache.TRANSLATION_CACHE_DIR_NAMES.values())
        or any(_BUILD_FOLDER_RE.fullmatch(child.name) for child in path.iterdir() if child.is_dir())
    ):
        raise CacheDirError(
            f"'{path}' does not look like a gt4py.next cache base directory: it is not named"
            f" '{config.BUILD_CACHE_DIR.name}' and contains neither a translation cache"
            f" ({', '.join(cache.TRANSLATION_CACHE_DIR_NAMES.values())}) nor a build cache folder."
        )
    return path


def get_translation_cache_dirs(
    cache_base: pathlib.Path, backends: Sequence[str]
) -> dict[str, pathlib.Path]:
    """Return the translation cache directory of each of `backends`."""
    return {
        backend: cache_base / cache.TRANSLATION_CACHE_DIR_NAMES[backend] for backend in backends
    }


def read_entry(path: pathlib.Path) -> stages.ProgramSource:
    """Unpickle one translation cache entry.

    Reads the file directly instead of going through `filecache.FileCache`, which
    deletes entries it fails to unpickle.
    """
    with path.open("rb") as fp:
        return pickle.load(fp)


def describe_entry(program_source: stages.ProgramSource) -> dict[str, Any]:
    """Return the facts worth reporting about one unpickled entry."""
    details: dict[str, Any] = {
        "program": program_source.entry_point.name,
        "parameters": len(program_source.entry_point.parameters),
        "source_language": program_source.code_spec.source_language,
    }
    source_code = program_source.source_code
    if isinstance(source_code, dict):
        # The DaCe backend stores the SDFG as a JSON object, so it can be
        # inspected without importing dace.
        type_counts = _count_json_types(source_code)
        details |= {
            "sdfg_name": source_code.get("attributes", {}).get("name"),
            "dace_version": source_code.get("dace_version"),
            "states": type_counts.get("SDFGState", 0),
            "maps": type_counts.get("MapEntry", 0),
        }
    else:
        details["source_characters"] = len(source_code)
    return details


def _count_json_types(obj: Any, counts: dict[str, int] | None = None) -> dict[str, int]:
    counts = {} if counts is None else counts
    if isinstance(obj, dict):
        if isinstance(type_name := obj.get("type"), str):
            counts[type_name] = counts.get(type_name, 0) + 1
        for value in obj.values():
            _count_json_types(value, counts)
    elif isinstance(obj, list):
        for value in obj:
            _count_json_types(value, counts)
    return counts


def find_entries(
    cache_base: pathlib.Path, backends: Sequence[str], *, program: str | None = None
) -> list[Entry]:
    """Collect the translation cache entries of `backends`, optionally filtered.

    Args:
        cache_base: Cache base directory. Missing cache directories are empty
            caches, not errors.
        backends: Backend names to scan.
        program: If given, keep only entries whose program name matches this glob.
            Unreadable entries have no program name and are dropped by any filter.
    """
    entries = []
    for backend, cache_dir in get_translation_cache_dirs(cache_base, backends).items():
        if not cache_dir.is_dir():
            continue
        for path in sorted(cache_dir.glob(f"*{ENTRY_SUFFIX}")):
            file_stat = path.stat()
            try:
                program_name: str | None = read_entry(path).entry_point.name
                error: str | None = None
            except Exception as e:
                program_name = None
                error = f"{type(e).__name__}: {e}"
            if program is not None and (
                program_name is None or not fnmatch.fnmatchcase(program_name, program)
            ):
                continue
            entries.append(
                Entry(
                    backend=backend,
                    key=path.stem,
                    path=path,
                    size=file_stat.st_size,
                    mtime=file_stat.st_mtime,
                    program=program_name,
                    error=error,
                )
            )
    return entries


def find_build_dirs(cache_base: pathlib.Path, *, program: str | None = None) -> list[BuildDir]:
    """Collect the build cache folders, optionally filtered by a program glob.

    The shared compiledb folder is named like a program's build folder but belongs
    to no program, so it is left out and never deleted along with one.
    """
    if not cache_base.is_dir():
        return []
    build_dirs = []
    for path in sorted(cache_base.iterdir()):
        if not path.is_dir() or not (match := _BUILD_FOLDER_RE.fullmatch(path.name)):
            continue
        name = match.group("name").removesuffix(cache.BINDINGS_NAME_SUFFIX)
        if name.startswith(compiledb.COMPILEDB_PROTOTYPE_NAME_PREFIX):
            continue
        if program is None or fnmatch.fnmatchcase(name, program):
            build_dirs.append(
                BuildDir(
                    program=name,
                    path=path,
                    version_id=match.group("version_id"),
                    complete=build_data.is_build_complete(path),
                )
            )
    return build_dirs


def get_status(
    cache_base: pathlib.Path, backends: Sequence[str], *, program: str | None = None
) -> tuple[list[ProgramStatus], list[Entry]]:
    """Report per program what the next run does with each cache.

    Returns:
        The status of every program with a translation cache entry or a build
        folder, sorted by name, and the entries that could not be read.
    """
    entries = find_entries(cache_base, backends, program=program)
    build_dirs = find_build_dirs(cache_base, program=program)

    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for entry in entries:
        if entry.program is not None:
            counts[entry.program][entry.backend] += 1
    build_counts = collections.Counter(build_dir.program for build_dir in build_dirs)
    usable_build_counts = collections.Counter(
        build_dir.program for build_dir in build_dirs if build_dir.usable
    )

    return [
        ProgramStatus(
            program=name,
            entries=counts.get(name, {}),
            build_dirs=build_counts.get(name, 0),
            usable_build_dirs=usable_build_counts.get(name, 0),
        )
        for name in sorted(counts.keys() | build_counts.keys())
    ], [entry for entry in entries if entry.program is None]


def delete_entries(entries: Sequence[Entry], cache_dirs: Sequence[pathlib.Path]) -> None:
    """Remove translation cache entries, holding the same lock as the runtime.

    The caches are shared between concurrent processes (e.g. MPI ranks), so an
    unlocked unlink can race a writer.

    Raises:
        CacheDirError: If an entry is not a cache entry inside `cache_dirs`.
    """
    for entry in entries:
        if entry.path.suffix != ENTRY_SUFFIX or entry.path.parent not in cache_dirs:
            raise CacheDirError(
                f"refusing to remove '{entry.path}': not a translation cache entry."
            )
        with locking.lock(entry.path):
            entry.path.unlink(missing_ok=True)


def delete_build_dirs(build_dirs: Sequence[BuildDir], cache_base: pathlib.Path) -> None:
    """Remove build cache folders, holding the same lock as the runtime.

    Raises:
        CacheDirError: If a folder is not a build cache folder of `cache_base`.
    """
    for build_dir in build_dirs:
        if build_dir.path.parent != cache_base or not _BUILD_FOLDER_RE.fullmatch(
            build_dir.path.name
        ):
            raise CacheDirError(f"refusing to remove '{build_dir.path}': not a build cache folder.")
        with locking.lock(build_dir.path):
            shutil.rmtree(build_dir.path)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    scaled = size / 1024
    if scaled < 1024:
        return f"{scaled:.1f} KiB"
    return f"{scaled / 1024:.1f} MiB"


def _format_mtime(mtime: float) -> str:
    return datetime.datetime.fromtimestamp(mtime).isoformat(sep=" ", timespec="seconds")


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip())


def _cmd_path(args: argparse.Namespace) -> int:
    cache_base = get_cache_base(args.cache_dir)
    print(f"lifetime:   {config.BUILD_CACHE_LIFETIME.name.lower()}")
    print(f"cache base: {cache_base}{'' if cache_base.is_dir() else '  (does not exist yet)'}")
    for backend, cache_dir in get_translation_cache_dirs(cache_base, args.backends).items():
        exists = "" if cache_dir.is_dir() else "  (does not exist yet)"
        print(f"{backend + ':':11} {cache_dir}{exists}")
    return EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    cache_base = get_cache_base(args.cache_dir)
    entries = find_entries(cache_base, args.backends, program=args.filter)
    sort_keys: dict[str, Callable[[Entry], Any]] = {
        "mtime": lambda e: e.mtime,
        "size": lambda e: e.size,
        "program": lambda e: (e.program or "", e.key),
        "key": lambda e: e.key,
    }
    entries.sort(key=sort_keys[args.sort])

    if args.json:
        print(
            json.dumps(
                [{**dataclasses.asdict(e), "path": str(e.path)} for e in entries],
                indent=2,
                default=str,
            )
        )
        return EXIT_OK

    if not entries:
        print("no entries")
        return EXIT_OK
    _print_table(
        ("BACKEND", "KEY", "PROGRAM", "SIZE", "MTIME"),
        [
            (
                e.backend,
                e.key,
                e.program if e.program is not None else "<unreadable>",
                _format_size(e.size),
                _format_mtime(e.mtime),
            )
            for e in entries
        ],
    )
    return EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    cache_base = get_cache_base(args.cache_dir)
    matches = [e for e in find_entries(cache_base, args.backends) if e.key == args.key]
    if not matches:
        print(f"error: no entry with key '{args.key}'", file=sys.stderr)
        return EXIT_NOTHING_DONE

    for entry in matches:
        details: dict[str, Any] = {
            "key": entry.key,
            "backend": entry.backend,
            "path": entry.path,
            "size": _format_size(entry.size),
            "mtime": _format_mtime(entry.mtime),
        }
        if entry.error is None:
            details |= describe_entry(read_entry(entry.path))
        else:
            details["error"] = f"unreadable payload ({entry.error})"
        width = max(len(name) for name in details) + 2
        for name, value in details.items():
            print(f"{name + ':':{width}}{value}")
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    cache_base = get_cache_base(args.cache_dir)
    statuses, unreadable = get_status(cache_base, args.backends, program=args.program)

    if not statuses:
        print("no cached programs")
    else:
        _print_table(
            ("PROGRAM", "ENTRIES", "BUILDS", "TRANSLATION", "BUILD"),
            [
                (
                    status.program,
                    ", ".join(f"{n} {backend}" for backend, n in sorted(status.entries.items()))
                    or "-",
                    f"{status.usable_build_dirs}"
                    if status.build_dirs == status.usable_build_dirs
                    else f"{status.usable_build_dirs} (+{status.build_dirs - status.usable_build_dirs} stale)",
                    "may replay" if status.replays else "will re-translate",
                    "will recompile" if status.recompiles else "may reuse",
                )
                for status in statuses
            ],
        )
    sys.stdout.flush()  # keep the table above the warnings when both go to a terminal
    if dangerous := [s for s in statuses if s.replays and s.recompiles]:
        print(
            f"\nWARNING: {len(dangerous)} program(s) will recompile and may replay a cached"
            " translation while doing so. The library then gets a fresh mtime although a changed"
            " transformation or pass never runs.",
            file=sys.stderr,
        )
    if stale := sum(s.build_dirs - s.usable_build_dirs for s in statuses):
        print(
            f"\nnote: {stale} build folder(s) cannot be hit by this environment, most of them"
            f" built under a build-cache version other than '{config.BUILD_CACHE_VERSION_ID}'."
            " The same version salts the translation cache, but entries do not record it, so"
            " entries left by those runs are still counted as REPLAY above although they"
            " cannot be hit either.",
            file=sys.stderr,
        )
    if unreadable:
        print(
            f"\nnote: {len(unreadable)} entr(ies) could not be read; the runtime discards"
            " those too, so they are not replayed.",
            file=sys.stderr,
        )

    replaying = [status for status in statuses if status.replays]
    if args.fail_if_cached and replaying:
        print(
            f"error: {len(replaying)} program(s) may replay a cached translation.",
            file=sys.stderr,
        )
        return EXIT_NOTHING_DONE
    return EXIT_OK


def _confirm(question: str) -> bool:
    """Ask a yes/no question on an interactive terminal, defaulting to no."""
    try:
        answer = input(f"{question} [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().casefold() in ("y", "yes")


def _cmd_delete(args: argparse.Namespace) -> int:
    cache_base = get_cache_base(args.cache_dir)
    cache_dirs = get_translation_cache_dirs(cache_base, args.backends)

    entries = find_entries(cache_base, args.backends, program=args.program)
    if args.key:
        entries = [entry for entry in entries if entry.key in args.key]

    build_dirs: list[BuildDir] = []
    if args.include_build_dirs:
        if args.key:
            programs = {entry.program for entry in entries if entry.program is not None}
            build_dirs = [b for b in find_build_dirs(cache_base) if b.program in programs]
        else:
            build_dirs = find_build_dirs(cache_base, program=args.program)

    if not entries and not build_dirs:
        print("error: nothing matched the given selector.", file=sys.stderr)
        return EXIT_NOTHING_DONE

    print(f"from {cache_base}:")
    for entry in entries:
        print(f"  {entry.path.relative_to(cache_base)} ({entry.program or '<unreadable>'})")
    for build_dir in build_dirs:
        print(f"  {build_dir.path.relative_to(cache_base)}/ [build]")
    summary = f"{len(entries)} entr(ies) and {len(build_dirs)} build folder(s)"

    if args.dry_run:
        print(f"\ndry run: nothing was removed ({summary} matched).")
        return EXIT_OK
    if not args.yes:
        if not sys.stdin.isatty():
            print(
                f"error: refusing to remove {summary} without a confirmation. Pass --yes to"
                " confirm, or --dry-run to preview.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        if not _confirm(f"\nRemove {summary}?"):
            print("aborted, nothing was removed.")
            return EXIT_NOTHING_DONE

    delete_entries(entries, list(cache_dirs.values()))
    delete_build_dirs(build_dirs, cache_base)
    print(f"\nremoved {summary}.")
    return EXIT_OK


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Cache base directory to operate on (default: the one the current environment uses).",
    )
    parser.add_argument(
        "--backend",
        dest="backends",
        choices=[*cache.TRANSLATION_CACHE_DIR_NAMES, "all"],
        default="all",
        help="Translation cache(s) to operate on (default: all).",
    )


def _make_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    path_parser = subparsers.add_parser("path", help="Print the resolved cache directories.")
    path_parser.set_defaults(func=_cmd_path)

    list_parser = subparsers.add_parser("list", help="List the translation cache entries.")
    list_parser.add_argument(
        "--filter", metavar="GLOB", default=None, help="Only entries whose program name matches."
    )
    list_parser.add_argument("--sort", choices=["mtime", "size", "program", "key"], default="mtime")
    list_parser.add_argument("--json", action="store_true", help="Print as JSON.")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser("show", help="Show the details of one entry.")
    show_parser.add_argument("key", help="Entry key, as printed by `list`.")
    show_parser.set_defaults(func=_cmd_show)

    status_parser = subparsers.add_parser(
        "status",
        help="Report what the next run does with each cache.",
        description=(
            "Report, per program, what the next run can do with each cache."
            " The verdicts are deliberately asymmetric, because only one direction can be"
            " known: 'will re-translate' and 'will recompile' are guarantees, since nothing"
            " on disk can serve that program; 'may replay' and 'may reuse' only say that"
            " something on disk could. Whether it is actually hit depends on a fingerprint"
            " taken at run time over the lowered program and its arguments, which this tool"
            " does not have. So a cached entry left over from an earlier version of the"
            " program source is reported as 'may replay' although editing that source"
            " already invalidated it. Build folders that are unfinished, or were built"
            " under another build-cache version, are counted as stale instead, because their"
            " name records the version; translation cache entries record none, so they"
            " cannot be told apart that way."
        ),
    )
    status_parser.add_argument(
        "--program", metavar="GLOB", default=None, help="Only programs whose name matches."
    )
    status_parser.add_argument(
        "--fail-if-cached",
        action="store_true",
        help="Exit with a non-zero status if any program would replay a cached translation.",
    )
    status_parser.set_defaults(func=_cmd_status)

    delete_parser = subparsers.add_parser(
        "delete",
        help="Remove translation cache entries.",
        description=(
            "List what the selector matches, then remove it after a confirmation. The"
            " confirmation is only asked for on an interactive terminal; elsewhere --yes is"
            " required, so a script never blocks on a prompt."
        ),
    )
    selectors = delete_parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--key", action="append", metavar="KEY", help="Entry key; repeatable.")
    selectors.add_argument("--program", metavar="GLOB", help="Entries whose program name matches.")
    selectors.add_argument("--all", action="store_true", help="All entries.")
    confirmation = delete_parser.add_mutually_exclusive_group()
    confirmation.add_argument(
        "--yes", "-y", action="store_true", help="Remove without asking for confirmation."
    )
    confirmation.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Only describe what would be removed, then exit.",
    )
    delete_parser.add_argument(
        "--include-build-dirs",
        action="store_true",
        help="Also remove the build cache folders of the selected programs.",
    )
    delete_parser.set_defaults(func=_cmd_delete)

    for subparser in (path_parser, list_parser, show_parser, status_parser, delete_parser):
        _add_common_arguments(subparser)
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Run the CLI. Entry point of the `gt4py-next-cache` command."""
    args = _make_parser(prog).parse_args(argv)
    args.backends = (
        list(cache.TRANSLATION_CACHE_DIR_NAMES) if args.backends == "all" else [args.backends]
    )
    try:
        return args.func(args)
    except CacheDirError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    # `python -m` leaves argparse with the module file as the program name.
    sys.exit(main(prog="python -m gt4py.next.gt_cache_manager"))
