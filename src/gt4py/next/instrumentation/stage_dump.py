# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Subscriber that dumps toolchain stage artifacts to disk (`GT4PY_DUMP_STAGES`).

Enabled automatically at import time of `gt4py.next.otf.workflow` when
`gt4py.next.config.DUMP_STAGES` is set; `enable` / `disable` are provided for
programmatic and test use (setting `config.DUMP_STAGES` after import requires
calling `enable`). Artifacts are grouped in one subdirectory per program; every
file gets an increasing index prefix, seeded from the files already in the
directory, so that the listing follows the pipeline order even when several
processes (e.g. compilation workers) write to it, and so that runs from
several processes or same-named programs never overwrite each other.

Two limitations of the current implementation:

- Only the environment variable dumps every stage. The compile pipeline runs in
  a worker process under the default `BUILD_JOBS_MODE=PROCESS`, and a worker
  registers this subscriber while importing `gt4py.next` — that is, from its own
  environment, before it receives the parent's configuration. A programmatic
  `config.DUMP_STAGES = ...; enable()` therefore only dumps the frontend stages;
  `enable` warns about it.
- Monolithic backends (e.g. `roundtrip`) have no `CompilePipeline` and
  execute in a single unnamed step, so only their frontend stages are dumped.
"""

from __future__ import annotations

import json
import pathlib
import re
import threading
import warnings
from typing import Any

from gt4py.next import config


#: Name the stage-dump subscriber is registered under on `stage_hook`.
SUBSCRIBER_NAME = "gt4py_dump_stages"

_index_lock = threading.Lock()
_next_index: dict[pathlib.Path, int] = {}
_index_prefix_pattern = re.compile(r"^(\d+)_")


def enable() -> None:
    """
    Register the stage-dump subscriber on `stage_hook` (idempotent).

    Warns when the compile pipeline runs in worker processes, since those
    register their own subscribers at import time and so never see a
    programmatic `config.DUMP_STAGES`.
    """
    from gt4py.next.otf import workflow

    if SUBSCRIBER_NAME in workflow.stage_hook.registry:
        return
    if config.DUMP_STAGES is not None:
        pathlib.Path(config.DUMP_STAGES).mkdir(parents=True, exist_ok=True)
    if config.BUILD_JOBS_MODE is not config.BuildJobsMode.SERIAL:
        warnings.warn(
            "Stage dumping was enabled programmatically while 'BUILD_JOBS_MODE' is"
            f" '{config.BUILD_JOBS_MODE.value}', so the compile pipeline runs in worker"
            " processes that do not see this subscriber: only the frontend stages will"
            " be dumped. Set the 'GT4PY_DUMP_STAGES' environment variable (which the"
            " workers do see) or 'GT4PY_BUILD_JOBS_MODE=serial' to dump every stage.",
            stacklevel=2,
        )
    workflow.stage_hook.register(dump_stage, name=SUBSCRIBER_NAME)


def disable() -> None:
    """Remove the stage-dump subscriber from `stage_hook` (idempotent)."""
    from gt4py.next.otf import workflow

    if SUBSCRIBER_NAME in workflow.stage_hook.registry:
        workflow.stage_hook.remove(SUBSCRIBER_NAME)


def dump_stage(name: str, artifact: Any) -> None:
    """
    Write a stage artifact under `config.DUMP_STAGES` (a `stage_hook` subscriber).

    Dumping is purely observational, so a failure to describe or write an
    artifact warns and is otherwise ignored: it must never turn a working
    compilation into a failing one (in the default `PROCESS` build-jobs mode the
    pipeline runs in a worker, where such an error would surface late and out of
    context, if at all).

    Args:
        name: Name of the pipeline step that produced `artifact`.
        artifact: The artifact to dump. Unknown artifact types fall back to
            their `repr`.
    """
    if (dump_dir := config.DUMP_STAGES) is None:
        return
    try:
        program_name, text, extension = _describe(artifact)
        target_dir = pathlib.Path(dump_dir) / _sanitize(program_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        _write_unique(target_dir, name, text, extension)
    except Exception as error:
        try:
            warnings.warn(
                f"Could not dump the artifact of stage '{name}'"
                f" ('{type(artifact).__name__}'): {error!s}.",
                stacklevel=2,
            )
        except Exception:
            # Warnings can be configured to raise (`-W error`), which would defeat
            # the whole point of not letting a failed dump break a compilation.
            pass


def _describe(artifact: Any) -> tuple[str, str, str]:
    """Best-effort (program_name, text, file_extension) for a stage artifact."""
    from gt4py.next.ffront import stages as ffront_stages
    from gt4py.next.iterator import ir as itir
    from gt4py.next.otf import artifacts, workflow

    match artifact:
        case workflow.ProgramWithArgs():
            return _describe(artifact.definition)
        case itir.Program():
            return str(artifact.id), str(artifact), "txt"  # `str` pretty-prints the IR
        case ffront_stages.DSLFieldOperatorDef() | ffront_stages.DSLProgramDef():
            return artifact.definition.__name__, _dsl_source(artifact.definition), "py"
        case ffront_stages.FOASTOperatorDef():
            return str(artifact.foast_node.id), _foast_text(artifact.foast_node), "txt"
        case ffront_stages.PASTProgramDef():
            # PAST has no pretty printer (unlike FOAST and ITIR), and eve nodes
            # define `__str__` as `__repr__`, so `repr` is all there is.
            return str(artifact.past_node.id), repr(artifact.past_node), "txt"
        case artifacts.ProgramSource():
            return (
                artifact.entry_point.name,
                _source_text(artifact.source_code),
                artifact.code_spec.file_extension,
            )
        case artifacts.ExtensionSource():
            program_name = artifact.program_source.entry_point.name
            if artifact.binding_source is not None:
                return (
                    program_name,
                    _source_text(artifact.binding_source.source_code),
                    _binding_extension(artifact.program_source.code_spec),
                )
            return program_name, "(no binding source generated)", "txt"
        case _:
            return _guess_name(artifact), repr(artifact), "txt"


def _binding_extension(code_spec: Any) -> str:
    """Guess the language of a `BindingSource`, which records none of its own.

    Bindings are written in the program's own language when that language can
    call into Python (the C++-like backends compile them into the extension
    module), and in Python otherwise — the dace backend, for instance, binds an
    SDFG with a generated Python function.
    """
    from gt4py.next.otf import artifacts

    return code_spec.file_extension if isinstance(code_spec, artifacts.CPPLikeCodeSpec) else "py"


def _source_text(source_code: Any) -> str:
    """Render a source container's code, which is not always text.

    The dace backend stores the SDFG as its deserialized JSON object, so a
    `ProgramSource.source_code` can be a `dict` despite its annotation.
    """
    if isinstance(source_code, str):
        return source_code
    return json.dumps(source_code, indent=2, default=repr)


def _dsl_source(definition: Any) -> str:
    import inspect

    try:
        return inspect.getsource(definition)
    except (OSError, TypeError):
        return repr(definition)


def _foast_text(node: Any) -> str:
    from gt4py.next.ffront import foast_pretty_printer

    try:
        return foast_pretty_printer.pretty_format(node)
    except Exception:
        return repr(node)


def _guess_name(artifact: Any) -> str:
    if (name := getattr(artifact, "entry_point_name", None)) is not None:  # CPP artifacts
        return str(name)
    if (path := getattr(artifact, "library_path", None)) is not None:  # dace artifacts
        return pathlib.Path(path).stem.removeprefix("lib")
    return "unknown_program"


def _sanitize(name: str) -> str:
    """Turn a program name into a single directory name below the dump root."""
    sanitized = re.sub(r"[^\w.-]", "_", name)
    # Names made of dots only ('.', '..') are path navigation rather than names
    # and would write outside the per-program directory. Program names are
    # usually Python identifiers, but `_guess_name` also derives them from
    # library file names.
    return sanitized if sanitized.strip(".") else "unknown_program"


def _initial_index(target_dir: pathlib.Path) -> int:
    """Continue numbering after the highest index prefix already in `target_dir`.

    The counter is per process, but under the default `BUILD_JOBS_MODE=PROCESS`
    the compile pipeline runs in a spawned worker whose counter starts over.
    Seeding it from the directory keeps a sorted listing in pipeline order.
    """
    highest = -1
    for path in target_dir.iterdir():
        if (match := _index_prefix_pattern.match(path.name)) is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def _write_unique(target_dir: pathlib.Path, stage_name: str, text: str, extension: str) -> None:
    with _index_lock:
        if (index := _next_index.get(target_dir)) is None:
            index = _initial_index(target_dir)
        while True:
            path = target_dir / f"{index:03d}_{stage_name}.{extension}"
            try:
                stream = path.open("x", encoding="utf-8")
            except FileExistsError:  # another process / earlier run owns this index
                index += 1
                continue
            try:
                with stream:
                    stream.write(text)
            except Exception:
                # Exclusive creation succeeded but writing did not: do not leave an
                # empty file behind, it would look like a dumped but empty stage.
                path.unlink(missing_ok=True)
                raise
            break
        _next_index[target_dir] = index + 1
