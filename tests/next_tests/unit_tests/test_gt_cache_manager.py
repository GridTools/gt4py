# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from gt4py._core import filecache
from gt4py.next import config, gt_cache_manager
from gt4py.next.otf import code_specs, stages
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import build_data, cache
from gt4py.next.otf.compilation.build_systems import compiledb


def make_program_source(name: str, backend: str) -> stages.ProgramSource:
    """Build a payload of the shape the translation step of `backend` caches."""
    if backend == "dace":
        code_spec = code_specs.SDFGCodeSpec()
        source_code = {"type": "SDFG", "attributes": {"name": name}, "nodes": []}
    else:
        code_spec = code_specs.CPPCodeSpec()
        source_code = f"// generated code of {name}"
    return stages.ProgramSource(
        entry_point=interface.Function(name, ()),
        source_code=source_code,
        library_deps=(),
        code_spec=code_spec,
    )


def write_entry(cache_base: pathlib.Path, backend: str, name: str, salt: str = "") -> str:
    """Write one translation cache entry through `FileCache`, as the runtime does."""
    file_cache = filecache.FileCache(cache_base / cache.TRANSLATION_CACHE_DIR_NAMES[backend])
    key = f"{backend}-{name}-{salt}"
    file_cache[key] = make_program_source(name, backend)
    return file_cache._get_path(key).stem


def write_build_dir(
    cache_base: pathlib.Path,
    name: str,
    salt: str = "0",
    *,
    complete: bool = True,
    version_id: str | None = None,
) -> pathlib.Path:
    """Create a folder named the way `cache.get_cache_folder` names them."""
    version_id = config.BUILD_CACHE_VERSION_ID if version_id is None else version_id
    folder = cache_base / f"{name}{cache.BINDINGS_NAME_SUFFIX}_{salt * 16}_{version_id}"
    folder.mkdir(parents=True)
    (folder / "libprogram.so").write_text("not really a library")
    if complete:
        (folder / build_data.COMPILE_COMPLETE_MARKER_NAME).touch()
    return folder


class FakeStdin:
    """Stand-in for `sys.stdin` that reports whether it is a terminal."""

    def __init__(self, interactive: bool) -> None:
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


@pytest.fixture
def cache_base(tmp_path):
    base = tmp_path / config.BUILD_CACHE_DIR.name
    base.mkdir()
    return base


def test_get_cache_base_follows_persistent_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BUILD_CACHE_DIR", tmp_path / ".gt4py_cache")
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.PERSISTENT)
    assert gt_cache_manager.get_cache_base() == tmp_path / ".gt4py_cache"


def test_get_cache_base_follows_session_config(monkeypatch):
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.SESSION)
    assert gt_cache_manager.get_cache_base() == cache.get_cache_base_path(
        config.BuildCacheLifetime.SESSION
    )


def test_get_cache_base_accepts_cache_like_dirs(tmp_path, cache_base):
    assert gt_cache_manager.get_cache_base(cache_base) == cache_base.resolve()

    by_translation_cache = tmp_path / "elsewhere"
    (by_translation_cache / cache.TRANSLATION_CACHE_DIR_NAMES["dace"]).mkdir(parents=True)
    assert gt_cache_manager.get_cache_base(by_translation_cache) == by_translation_cache.resolve()

    by_build_dir = tmp_path / "somewhere"
    by_build_dir.mkdir()
    write_build_dir(by_build_dir, "prog")
    assert gt_cache_manager.get_cache_base(by_build_dir) == by_build_dir.resolve()


def test_get_cache_base_rejects_other_dirs(tmp_path):
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "notes.txt").write_text("hello")
    with pytest.raises(gt_cache_manager.CacheDirError, match="does not look like"):
        gt_cache_manager.get_cache_base(tmp_path / "unrelated")

    with pytest.raises(gt_cache_manager.CacheDirError, match="not a directory"):
        gt_cache_manager.get_cache_base(tmp_path / "missing")


def test_missing_cache_dirs_are_an_empty_cache(cache_base):
    assert gt_cache_manager.find_entries(cache_base, ["dace", "gtfn"]) == []
    assert gt_cache_manager.find_build_dirs(cache_base) == []
    assert gt_cache_manager.get_status(cache_base, ["dace", "gtfn"]) == ([], [])


def test_find_entries_reads_program_and_backend(cache_base):
    dace_key = write_entry(cache_base, "dace", "foo")
    gtfn_key = write_entry(cache_base, "gtfn", "bar")

    entries = gt_cache_manager.find_entries(cache_base, ["dace", "gtfn"])

    assert {(e.backend, e.key, e.program) for e in entries} == {
        ("dace", dace_key, "foo"),
        ("gtfn", gtfn_key, "bar"),
    }
    assert all(e.error is None and e.size > 0 for e in entries)


def test_find_entries_filters_by_program_and_backend(cache_base):
    write_entry(cache_base, "dace", "foo")
    write_entry(cache_base, "dace", "foobar")
    write_entry(cache_base, "gtfn", "foo")

    assert len(gt_cache_manager.find_entries(cache_base, ["dace"], program="foo*")) == 2
    assert len(gt_cache_manager.find_entries(cache_base, ["dace", "gtfn"], program="foo")) == 2
    assert gt_cache_manager.find_entries(cache_base, ["gtfn"], program="nope") == []


def test_find_entries_ignores_lock_files(cache_base):
    key = write_entry(cache_base, "dace", "foo")
    cache_dir = cache_base / cache.TRANSLATION_CACHE_DIR_NAMES["dace"]
    # A writer killed mid-write leaves its lock file behind.
    (cache_dir / f"{key}.filelock_FileLock.lock").touch()

    assert len(gt_cache_manager.find_entries(cache_base, ["dace"])) == 1


def test_find_build_dirs_recovers_program_name(cache_base):
    write_build_dir(cache_base, "foo")
    (cache_base / "not-a-build-dir").mkdir()

    build_dirs = gt_cache_manager.find_build_dirs(cache_base)

    assert [b.program for b in build_dirs] == ["foo"]
    assert gt_cache_manager.find_build_dirs(cache_base, program="ba*") == []


def test_find_build_dirs_skips_the_shared_compiledb(cache_base):
    write_build_dir(cache_base, "foo")
    compiledb_dir = write_build_dir(
        cache_base, f"{compiledb.COMPILEDB_PROTOTYPE_NAME_PREFIX}_gridtools_cpu_Release", salt="1"
    )

    assert [b.program for b in gt_cache_manager.find_build_dirs(cache_base)] == ["foo"]

    gt_cache_manager.main(
        ["delete", "--all", "--include-build-dirs", "--yes", "--cache-dir", str(cache_base)]
    )

    assert compiledb_dir.is_dir()


def test_corrupt_entry_degrades_instead_of_crashing(cache_base, capsys):
    write_entry(cache_base, "dace", "foo")
    corrupt = cache_base / cache.TRANSLATION_CACHE_DIR_NAMES["dace"] / "deadbeefdeadbeef.pkl"
    corrupt.write_bytes(b"not a pickle")

    entries = gt_cache_manager.find_entries(cache_base, ["dace"])
    corrupt_entries = [e for e in entries if e.key == corrupt.stem]

    assert len(entries) == 2
    assert corrupt_entries[0].program is None
    assert corrupt_entries[0].error
    assert corrupt.exists()  # inspection must not delete what it cannot read

    assert gt_cache_manager.main(["list", "--cache-dir", str(cache_base)]) == 0
    assert "<unreadable>" in capsys.readouterr().out

    assert gt_cache_manager.main(["status", "--cache-dir", str(cache_base)]) == 0
    assert "could not be read" in capsys.readouterr().err


def test_status_reports_replay_and_retranslate(cache_base):
    write_entry(cache_base, "dace", "cached")
    write_entry(cache_base, "dace", "cached", salt="second")
    write_entry(cache_base, "gtfn", "cached")
    write_build_dir(cache_base, "built_only")

    statuses, unreadable = gt_cache_manager.get_status(cache_base, ["dace", "gtfn"])

    assert unreadable == []
    assert [s.program for s in statuses] == ["built_only", "cached"]
    assert statuses[0].replays is False
    assert statuses[0].build_dirs == 1
    assert statuses[1].replays is True
    assert statuses[1].entries == {"dace": 2, "gtfn": 1}
    assert statuses[1].build_dirs == 0


def test_status_reports_the_two_caches_independently(cache_base):
    write_entry(cache_base, "dace", "warm")
    write_build_dir(cache_base, "warm")
    write_entry(cache_base, "dace", "translated_only")
    write_build_dir(cache_base, "built_only")

    statuses = {s.program: s for s in gt_cache_manager.get_status(cache_base, ["dace"])[0]}

    # both caches warm: nothing is redone, which is why the two verdicts are separate
    assert (statuses["warm"].replays, statuses["warm"].recompiles) == (True, False)
    # the trap: the library is rebuilt, the translation is not
    assert (statuses["translated_only"].replays, statuses["translated_only"].recompiles) == (
        True,
        True,
    )
    assert (statuses["built_only"].replays, statuses["built_only"].recompiles) == (False, False)


def test_status_counts_a_build_from_another_version_as_recompile(cache_base):
    write_entry(cache_base, "dace", "foo")
    write_build_dir(cache_base, "foo", version_id="0.0.1+ancient")

    (status,) = gt_cache_manager.get_status(cache_base, ["dace"])[0]

    assert status.build_dirs == 1
    assert status.usable_build_dirs == 0
    assert status.recompiles is True


def test_status_notes_that_entries_may_be_stale_too(cache_base, capsys):
    write_entry(cache_base, "dace", "foo")
    write_build_dir(cache_base, "foo", version_id="0.0.1+ancient")

    assert gt_cache_manager.main(["status", "--cache-dir", str(cache_base)]) == 0

    assert "cannot be hit either" in capsys.readouterr().err


def test_status_counts_an_unfinished_build_as_recompile(cache_base):
    write_entry(cache_base, "dace", "foo")
    write_build_dir(cache_base, "foo", complete=False)

    (status,) = gt_cache_manager.get_status(cache_base, ["dace"])[0]

    assert status.build_dirs == 1
    assert status.usable_build_dirs == 0
    assert status.recompiles is True


def test_status_accepts_build_data_as_finished(cache_base):
    folder = write_build_dir(cache_base, "foo", complete=False)
    build_data.write_data(
        build_data.BuildData(
            status=build_data.BuildStatus.COMPILED,
            module=folder / "libprogram.so",
            entry_point_name="foo",
        ),
        folder,
    )

    (status,) = gt_cache_manager.get_status(cache_base, ["gtfn"])[0]

    assert status.recompiles is False


def test_status_warns_about_replay_without_build_dir(cache_base, capsys):
    write_entry(cache_base, "dace", "foo")

    assert gt_cache_manager.main(["status", "--cache-dir", str(cache_base)]) == 0

    captured = capsys.readouterr()
    assert "REPLAY" in captured.out
    assert "recompile" in captured.out
    assert "does NOT run" in captured.err


def test_status_does_not_warn_when_both_caches_are_warm(cache_base, capsys):
    write_entry(cache_base, "dace", "foo")
    write_build_dir(cache_base, "foo")

    assert gt_cache_manager.main(["status", "--cache-dir", str(cache_base)]) == 0

    captured = capsys.readouterr()
    assert "REPLAY" in captured.out
    assert "reuse" in captured.out
    assert "WARNING" not in captured.err


def test_status_fail_if_cached(cache_base):
    write_build_dir(cache_base, "foo")
    argv = ["status", "--cache-dir", str(cache_base), "--fail-if-cached"]

    assert gt_cache_manager.main(argv) == 0

    write_entry(cache_base, "dace", "foo")

    assert gt_cache_manager.main(argv) == gt_cache_manager.EXIT_NOTHING_DONE


def test_status_program_filter(cache_base, capsys):
    write_entry(cache_base, "dace", "foo")
    write_entry(cache_base, "dace", "bar")

    gt_cache_manager.main(["status", "--cache-dir", str(cache_base), "--program", "fo*"])

    captured = capsys.readouterr()
    assert "foo" in captured.out
    assert "bar" not in captured.out


def test_show_reports_payload_details(cache_base, capsys):
    key = write_entry(cache_base, "dace", "foo")

    assert gt_cache_manager.main(["show", key, "--cache-dir", str(cache_base)]) == 0

    out = capsys.readouterr().out
    assert "program:" in out and "foo" in out
    assert "SDFG" in out


def test_show_unknown_key(cache_base):
    assert (
        gt_cache_manager.main(["show", "0123456789abcdef", "--cache-dir", str(cache_base)])
        == gt_cache_manager.EXIT_NOTHING_DONE
    )


def test_delete_dry_run_keeps_entries(cache_base, capsys):
    write_entry(cache_base, "dace", "foo")

    assert (
        gt_cache_manager.main(["delete", "--all", "--dry-run", "--cache-dir", str(cache_base)]) == 0
    )

    assert "dry run" in capsys.readouterr().out
    assert len(gt_cache_manager.find_entries(cache_base, ["dace"])) == 1


def test_delete_asks_before_removing_on_a_terminal(cache_base, monkeypatch, capsys):
    write_entry(cache_base, "dace", "foo")
    monkeypatch.setattr(gt_cache_manager.sys, "stdin", FakeStdin(interactive=True))
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    assert gt_cache_manager.main(["delete", "--all", "--cache-dir", str(cache_base)]) == 0

    assert gt_cache_manager.find_entries(cache_base, ["dace"]) == []


@pytest.mark.parametrize("answer", ["n", "", "no", "whatever"])
def test_delete_declined_removes_nothing(cache_base, monkeypatch, answer):
    write_entry(cache_base, "dace", "foo")
    monkeypatch.setattr(gt_cache_manager.sys, "stdin", FakeStdin(interactive=True))
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)

    result = gt_cache_manager.main(["delete", "--all", "--cache-dir", str(cache_base)])

    assert result == gt_cache_manager.EXIT_NOTHING_DONE
    assert len(gt_cache_manager.find_entries(cache_base, ["dace"])) == 1


def test_delete_never_prompts_without_a_terminal(cache_base, monkeypatch, capsys):
    write_entry(cache_base, "dace", "foo")
    monkeypatch.setattr(gt_cache_manager.sys, "stdin", FakeStdin(interactive=False))

    def _no_prompting(prompt=""):
        raise AssertionError("a non-interactive run must not block on a prompt")

    monkeypatch.setattr("builtins.input", _no_prompting)

    result = gt_cache_manager.main(["delete", "--all", "--cache-dir", str(cache_base)])

    assert result == gt_cache_manager.EXIT_ERROR
    assert "--yes" in capsys.readouterr().err
    assert len(gt_cache_manager.find_entries(cache_base, ["dace"])) == 1


def test_delete_declined_at_eof_removes_nothing(cache_base, monkeypatch):
    write_entry(cache_base, "dace", "foo")
    monkeypatch.setattr(gt_cache_manager.sys, "stdin", FakeStdin(interactive=True))

    def _eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)

    assert (
        gt_cache_manager.main(["delete", "--all", "--cache-dir", str(cache_base)])
        == gt_cache_manager.EXIT_NOTHING_DONE
    )
    assert len(gt_cache_manager.find_entries(cache_base, ["dace"])) == 1


def test_delete_removes_selected_entries(cache_base):
    write_entry(cache_base, "dace", "foo")
    write_entry(cache_base, "dace", "bar")

    assert (
        gt_cache_manager.main(
            ["delete", "--program", "foo", "--yes", "--cache-dir", str(cache_base)]
        )
        == 0
    )

    assert [e.program for e in gt_cache_manager.find_entries(cache_base, ["dace"])] == ["bar"]


def test_delete_by_key(cache_base):
    key = write_entry(cache_base, "dace", "foo")
    write_entry(cache_base, "dace", "bar")

    assert (
        gt_cache_manager.main(["delete", "--key", key, "--yes", "--cache-dir", str(cache_base)])
        == 0
    )

    assert [e.program for e in gt_cache_manager.find_entries(cache_base, ["dace"])] == ["bar"]


def test_delete_without_match_exits_non_zero(cache_base):
    write_entry(cache_base, "dace", "foo")

    assert (
        gt_cache_manager.main(
            ["delete", "--program", "nope", "--yes", "--cache-dir", str(cache_base)]
        )
        == gt_cache_manager.EXIT_NOTHING_DONE
    )


def test_delete_leaves_build_dirs_alone(cache_base):
    write_entry(cache_base, "dace", "foo")
    build_dir = write_build_dir(cache_base, "foo")

    assert gt_cache_manager.main(["delete", "--all", "--yes", "--cache-dir", str(cache_base)]) == 0

    assert gt_cache_manager.find_entries(cache_base, ["dace"]) == []
    assert build_dir.is_dir()


def test_delete_include_build_dirs(cache_base):
    write_entry(cache_base, "dace", "foo")
    foo_build_dir = write_build_dir(cache_base, "foo")
    bar_build_dir = write_build_dir(cache_base, "bar")

    assert (
        gt_cache_manager.main(
            [
                "delete",
                "--program",
                "foo",
                "--include-build-dirs",
                "--yes",
                "--cache-dir",
                str(cache_base),
            ]
        )
        == 0
    )

    assert gt_cache_manager.find_entries(cache_base, ["dace"]) == []
    assert not foo_build_dir.exists()
    assert bar_build_dir.is_dir()


def test_delete_backend_selection(cache_base):
    write_entry(cache_base, "dace", "foo")
    write_entry(cache_base, "gtfn", "foo")

    assert (
        gt_cache_manager.main(
            ["delete", "--all", "--yes", "--backend", "dace", "--cache-dir", str(cache_base)]
        )
        == 0
    )

    assert gt_cache_manager.find_entries(cache_base, ["dace"]) == []
    assert len(gt_cache_manager.find_entries(cache_base, ["gtfn"])) == 1


def test_delete_refuses_paths_outside_the_cache(cache_base, tmp_path):
    outsider = tmp_path / "precious.pkl"
    outsider.write_text("keep me")
    entry = gt_cache_manager.Entry(
        backend="dace",
        key=outsider.stem,
        path=outsider,
        size=0,
        mtime=0.0,
        program="foo",
        error=None,
    )

    with pytest.raises(gt_cache_manager.CacheDirError, match="refusing to remove"):
        gt_cache_manager.delete_entries(
            [entry], [cache_base / cache.TRANSLATION_CACHE_DIR_NAMES["dace"]]
        )

    assert outsider.exists()


def test_delete_refuses_build_dirs_outside_the_cache(cache_base, tmp_path):
    outsider = tmp_path / "precious"
    outsider.mkdir()
    build_dir = gt_cache_manager.BuildDir(
        program="foo",
        path=outsider,
        version_id=config.BUILD_CACHE_VERSION_ID,
        complete=True,
    )

    with pytest.raises(gt_cache_manager.CacheDirError, match="refusing to remove"):
        gt_cache_manager.delete_build_dirs([build_dir], cache_base)

    assert outsider.is_dir()


def test_path_command_reports_both_caches(cache_base, capsys):
    assert gt_cache_manager.main(["path", "--cache-dir", str(cache_base)]) == 0

    out = capsys.readouterr().out
    assert str(cache_base.resolve()) in out
    for dir_name in cache.TRANSLATION_CACHE_DIR_NAMES.values():
        assert dir_name in out


def test_list_json_output(cache_base, capsys):
    import json

    key = write_entry(cache_base, "dace", "foo")

    assert gt_cache_manager.main(["list", "--json", "--cache-dir", str(cache_base)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [(e["key"], e["program"], e["backend"]) for e in payload] == [(key, "foo", "dace")]


def test_console_script_points_at_main():
    import importlib.metadata

    entry_points = importlib.metadata.distribution("gt4py").entry_points
    console_scripts = {ep.name: ep for ep in entry_points if ep.group == "console_scripts"}

    assert "gt4py-next-cache" in console_scripts
    assert console_scripts["gt4py-next-cache"].load() is gt_cache_manager.main


def test_cli_reports_cache_dir_error(tmp_path, capsys):
    (tmp_path / "unrelated").mkdir()

    assert (
        gt_cache_manager.main(["list", "--cache-dir", str(tmp_path / "unrelated")])
        == gt_cache_manager.EXIT_ERROR
    )
    assert "does not look like" in capsys.readouterr().err
