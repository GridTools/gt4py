# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import shutil
import subprocess
from typing import Optional, TypeVar

from flufl import lock

from gt4py.next import config, errors
from gt4py.next.otf import languages, stages
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import build_data, cache, compiler
from gt4py.next.otf.compilation.build_systems import cmake


SrcL = TypeVar("SrcL", bound=languages.NanobindSrcL)


@dataclasses.dataclass
class CompiledbFactory(
    compiler.BuildSystemProjectGenerator[
        SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    """
    Create a CompiledbProject from a ``CompilableSource`` stage object with given CMake settings.

    Use CMake to generate a compiledb with the required sequence of build commands.
    Generate a compiledb only if there isn't one for the given combination of cmake configuration
    and library dependencies.
    """

    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG
    cmake_extra_flags: list[str] = dataclasses.field(default_factory=list)
    renew_compiledb: bool = False

    def __call__(
        self,
        source: stages.CompilableSource[
            SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python
        ],
        cache_lifetime: config.BuildCacheLifetime,
    ) -> CompiledbProject:
        if not source.binding_source:
            raise NotImplementedError(
                "Compiledb build system project requires separate bindings code file."
            )
        name = source.program_source.entry_point.name
        header_name = f"{name}.{source.program_source.language_settings.header_extension}"
        bindings_name = f"{name}_bindings.{source.program_source.language_settings.file_extension}"

        cc_prototype_program_source = _cc_prototype_program_source(
            deps=source.library_deps,
            build_type=self.cmake_build_type,
            cmake_flags=self.cmake_extra_flags or [],
            language=source.program_source.language,
            language_settings=source.program_source.language_settings,
            implicit_domain=source.program_source.implicit_domain,
        )

        compiledb_template = _cc_get_compiledb(
            self.renew_compiledb,
            cc_prototype_program_source,
            build_type=self.cmake_build_type,
            cmake_flags=self.cmake_extra_flags or [],
            cache_lifetime=cache_lifetime,
        )

        return CompiledbProject(
            root_path=cache.get_cache_folder(source, cache_lifetime),
            program_name=name,
            source_files={
                header_name: source.program_source.source_code,
                bindings_name: source.binding_source.source_code,
            },
            bindings_file_name=bindings_name,
            compile_commands_cache=compiledb_template,
        )


@dataclasses.dataclass()
class CompiledbProject(
    stages.BuildSystemProject[SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python]
):
    """
    Compiledb build system for gt4py programs.

    Rely on a pre-configured compiledb to run the right build steps in the right order.
    The advantage is that overall build time grows linearly in number of distinct configurations
    and not in number of GT4Py programs. In cases where many programs can reuse the same configuration,
    this can save multiple seconds per program over rerunning CMake configuration each time.

    Works independently of what is used to generate the compiledb.
    """

    root_path: pathlib.Path
    source_files: dict[str, str]
    program_name: str
    compile_commands_cache: pathlib.Path
    bindings_file_name: str

    def build(self) -> None:
        self._write_files()
        current_data = build_data.read_data(self.root_path)
        if current_data is None or current_data.status < build_data.BuildStatus.CONFIGURED:
            self._run_config()
            current_data = build_data.read_data(self.root_path)  # update after config
        if (
            current_data is not None
            and build_data.BuildStatus.CONFIGURED
            <= current_data.status
            < build_data.BuildStatus.COMPILED
        ):
            self._run_build()

    def _write_files(self) -> None:
        def ignore_function(folder: str, children: list[str]) -> list[str]:
            pattern = r"((lib.*\.a)|(.*\.lib))"
            folder_path = pathlib.Path(folder)

            ignored = []
            for child in children:
                if re.match(pattern, child):  # static library -> keep
                    continue
                if (folder_path / child).is_dir():  # folder -> keep
                    continue
                ignored.append(child)

            return ignored

        shutil.copytree(
            self.compile_commands_cache.parent,
            self.root_path,
            ignore=ignore_function,
            dirs_exist_ok=True,
        )

        for name, content in self.source_files.items():
            (self.root_path / name).write_text(content, encoding="utf-8")

        build_data.write_data(
            data=build_data.BuildData(
                status=build_data.BuildStatus.INITIALIZED,
                module=pathlib.Path(""),
                entry_point_name=self.program_name,
            ),
            path=self.root_path,
        )

    def _run_config(self) -> None:
        compile_db = json.loads(self.compile_commands_cache.read_text())

        (self.root_path / "build").mkdir(exist_ok=True)
        (self.root_path / "build" / "bin").mkdir(exist_ok=True)

        for entry in compile_db:
            for key, value in entry.items():
                entry[key] = (
                    value.replace("$NAME", self.program_name)
                    .replace("$BINDINGS_FILE", self.bindings_file_name)
                    .replace("$SRC_PATH", str(self.root_path))
                )

        (self.root_path / "compile_commands.json").write_text(json.dumps(compile_db))

        (build_script_path := self.root_path / "build.sh").write_text(
            "\n".join(["#!/bin/sh", "cd build", *(entry["command"] for entry in compile_db)])
        )
        try:
            build_script_path.chmod(0o755)
        except OSError:
            # if setting permissions fails, it's not a problem
            pass

        build_data.write_data(
            build_data.BuildData(
                status=build_data.BuildStatus.CONFIGURED,
                module=pathlib.Path(compile_db[-1]["directory"]) / compile_db[-1]["output"],
                entry_point_name=self.program_name,
            ),
            self.root_path,
        )

    def _run_build(self) -> None:
        logfile = self.root_path / "log_build.txt"
        compile_db = json.loads((self.root_path / "compile_commands.json").read_text())
        assert compile_db
        try:
            with logfile.open(mode="w") as log_file_pointer:
                for entry in compile_db:
                    log_file_pointer.write(entry["command"] + "\n")
                    log_file_pointer.flush()
                    subprocess.check_call(
                        entry["command"],
                        cwd=entry["directory"],
                        shell=True,
                        stdout=log_file_pointer,
                        stderr=log_file_pointer,
                    )
        except subprocess.CalledProcessError as e:
            with logfile.open(mode="r") as log_file_pointer:
                log = log_file_pointer.read()
            raise errors.CompilationError(log) from e

        build_data.update_status(new_status=build_data.BuildStatus.COMPILED, path=self.root_path)


def _cc_prototype_program_name(
    deps: tuple[interface.LibraryDependency, ...], build_type: str, flags: list[str]
) -> str:
    base_name = "compile_commands_cache"
    deps_str = "_".join(f"{dep.name}_{dep.version}" for dep in deps)
    flags_str = "_".join(re.sub(r"\W+", "", f) for f in flags)
    return "_".join([base_name, deps_str, build_type, flags_str]).replace(".", "_")


def _cc_prototype_program_source(
    deps: tuple[interface.LibraryDependency, ...],
    build_type: config.CMakeBuildType,
    cmake_flags: list[str],
    language: type[SrcL],
    language_settings: languages.LanguageWithHeaderFilesSettings,
    implicit_domain: bool,
) -> stages.ProgramSource:
    name = _cc_prototype_program_name(deps, build_type.value, cmake_flags)
    return stages.ProgramSource(
        entry_point=interface.Function(name=name, parameters=()),
        source_code="",
        library_deps=deps,
        language=language,
        language_settings=language_settings,
        implicit_domain=implicit_domain,
    )


def _cc_get_compiledb(
    renew_compiledb: bool,
    prototype_program_source: stages.ProgramSource,
    build_type: config.CMakeBuildType,
    cmake_flags: list[str],
    cache_lifetime: config.BuildCacheLifetime,
) -> pathlib.Path:
    cache_path = cache.get_cache_folder(
        stages.CompilableSource(prototype_program_source, None), cache_lifetime
    )

    # In a multi-threaded environment, multiple threads may try to create the compiledb at the same time
    # leading to compilation errors.
    with lock.Lock(str(cache_path / "compiledb.lock"), lifetime=120):  # type: ignore[attr-defined] # mypy not smart enough to understand custom export logic
        if renew_compiledb or not (compiled_db := _cc_find_compiledb(path=cache_path)):
            compiled_db = _cc_create_compiledb(
                prototype_program_source=prototype_program_source,
                build_type=build_type,
                cmake_flags=cmake_flags,
                cache_lifetime=cache_lifetime,
            )

            assert compiled_db.parent == cache_path

        return compiled_db


def _cc_find_compiledb(path: pathlib.Path) -> Optional[pathlib.Path]:
    compile_db_path = path / "compile_commands.json"
    if compile_db_path.exists():
        return compile_db_path
    return None


def _cc_create_compiledb(
    prototype_program_source: stages.ProgramSource,
    build_type: config.CMakeBuildType,
    cmake_flags: list[str],
    cache_lifetime: config.BuildCacheLifetime,
) -> pathlib.Path:
    prototype_project = cmake.CMakeFactory(
        cmake_generator_name="Ninja",
        cmake_build_type=build_type,
        cmake_extra_flags=cmake_flags,
    )(
        stages.CompilableSource(
            prototype_program_source, stages.BindingSource(source_code="", library_deps=())
        ),
        cache_lifetime,
    )

    path = prototype_project.root_path
    name = prototype_project.program_name
    binding_src_name = next(
        name
        for name in prototype_project.source_files.keys()
        if name.endswith(f"_bindings.{prototype_program_source.language_settings.file_extension}")
    )

    prototype_project.build()

    log_file = path / "log_compiledb.txt"

    with log_file.open("w") as log_file_pointer:
        commands_json_str = subprocess.check_output(
            ["ninja", "-t", "compdb"], cwd=path / "build", stderr=log_file_pointer
        ).decode("utf-8")
        commands = json.loads(commands_json_str)

    compile_db = [
        cmd for cmd in commands if name in pathlib.Path(cmd["file"]).stem and cmd["command"]
    ]

    assert compile_db

    for entry in compile_db:
        entry["directory"] = entry["directory"].replace(str(path), "$SRC_PATH")
        entry["command"] = (
            entry["command"]
            .replace(f"CMakeFiles/{name}.dir", ".")
            .replace(str(path), "$SRC_PATH")
            .replace(binding_src_name, "$BINDINGS_FILE")
            .replace(name, "$NAME")
            .replace("-I$SRC_PATH/build/_deps", f"-I{path}/build/_deps")
        )
        entry["file"] = (
            entry["file"]
            .replace(f"CMakeFiles/{name}.dir", ".")
            .replace(str(path), "$SRC_PATH")
            .replace(binding_src_name, "$BINDINGS_FILE")
        )
        entry["output"] = (
            entry["output"]
            .replace(f"CMakeFiles/{name}.dir", ".")
            .replace(binding_src_name, "$BINDINGS_FILE")
            .replace(name, "$NAME")
        )

    compile_db_path = path / "compile_commands.json"
    compile_db_path.write_text(json.dumps(compile_db))

    return compile_db_path
