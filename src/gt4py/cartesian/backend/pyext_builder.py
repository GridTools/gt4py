# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import contextlib
import copy
import distutils
import distutils.sysconfig
import io
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Type, Union, overload

import pybind11
import setuptools
from setuptools.command.build_ext import build_ext

from gt4py.cartesian import config as gt_config


def get_dace_module_path() -> Optional[str]:
    try:
        import dace

        return os.path.dirname(dace.__file__)
    except ImportError:
        return None


def get_cuda_compute_capability():
    try:
        import cupy as cp

        return cp.cuda.Device(0).compute_capability
    except ImportError:
        return None


def get_gt_pyext_build_opts(
    *,
    debug_mode: bool = False,
    add_profile_info: bool = False,
    uses_openmp: bool = True,
    uses_cuda: bool = False,
    gt_version: int = 1,
) -> Dict[str, Union[str, List[str], Dict[str, Any]]]:

    include_dirs = [gt_config.build_settings["boost_include_path"]]
    extra_compile_args_from_config = gt_config.build_settings["extra_compile_args"]

    if uses_cuda:
        compute_capability = get_cuda_compute_capability()
        cuda_arch = gt_config.build_settings["cuda_arch"] or compute_capability
        if not cuda_arch:
            raise RuntimeError("CUDA architecture could not be determined")
        if cuda_arch.startswith("sm_"):
            cuda_arch = cuda_arch.replace("sm_", "")
        if compute_capability and int(compute_capability) < int(cuda_arch):
            raise RuntimeError(
                f"CUDA architecture {cuda_arch} exceeds compute capability {compute_capability}"
            )
    else:
        cuda_arch = ""

    gt_include_path = gt_config.build_settings["gt_include_path"]

    import os

    extra_compile_args = dict(
        cxx=[
            "-std=c++17",
            "-ftemplate-depth={}".format(gt_config.build_settings["cpp_template_depth"]),
            "-fvisibility=hidden",
            "-fPIC",
            "-isystem{}".format(gt_include_path),
            "-isystem{}".format(gt_config.build_settings["boost_include_path"]),
            "-DBOOST_PP_VARIADICS",
            *extra_compile_args_from_config["cxx"],
        ],
        nvcc=[
            "-std=c++17",
            "-ftemplate-depth={}".format(gt_config.build_settings["cpp_template_depth"]),
            "-arch=sm_{}".format(cuda_arch),
            "-isystem={}".format(gt_include_path),
            "-isystem={}".format(gt_config.build_settings["boost_include_path"]),
            "-DBOOST_PP_VARIADICS",
            "-DBOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL",
            "-DBOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE",
            "--expt-relaxed-constexpr",
            "--compiler-options",
            "-fvisibility=hidden",
            "--compiler-options",
            "-fPIC",
            *extra_compile_args_from_config["nvcc"],
        ],
    )
    extra_link_args = gt_config.build_settings["extra_link_args"]

    mode_flags = ["-O0", "-ggdb"] if debug_mode else ["-O3", "-DNDEBUG"]
    extra_compile_args["cxx"].extend(mode_flags)
    extra_compile_args["nvcc"].extend(mode_flags)
    extra_link_args.extend(mode_flags)

    if dace_path := get_dace_module_path():
        extra_compile_args["cxx"].append(
            "-isystem{}".format(os.path.join(dace_path, "runtime/include"))
        )
        extra_compile_args["nvcc"].append(
            "-isystem={}".format(os.path.join(dace_path, "runtime/include"))
        )

    if add_profile_info:
        profile_flags = ["-pg"]
        extra_compile_args["cxx"].extend(profile_flags)
        extra_compile_args["nvcc"].extend(profile_flags)
        extra_link_args.extend(profile_flags)

    if uses_cuda:
        build_opts = dict(
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    else:
        build_opts = dict(
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args["cxx"],
            extra_link_args=extra_link_args,
        )

    if uses_openmp:
        cpp_flags = gt_config.build_settings["openmp_cppflags"]
        if uses_cuda:
            cuda_flags = []
            for cpp_flag in cpp_flags:
                cuda_flags.extend(["--compiler-options", cpp_flag])
            build_opts["extra_compile_args"]["nvcc"].extend(cuda_flags)
        elif cpp_flags:
            build_opts["extra_compile_args"].extend(cpp_flags)

        ld_flags = gt_config.build_settings["openmp_ldflags"]
        if ld_flags:
            build_opts["extra_link_args"].extend(ld_flags)

    return build_opts


# The following tells mypy to accept unpacking kwargs
@overload
def build_pybind_ext(
    name: str,
    sources: list,
    build_path: str,
    target_path: str,
    **kwargs: str,
) -> Tuple[str, str]:
    pass


def build_pybind_ext(
    name: str,
    sources: list,
    build_path: str,
    target_path: str,
    *,
    include_dirs: Optional[List[str]] = None,
    library_dirs: Optional[List[str]] = None,
    libraries: Optional[List[str]] = None,
    extra_compile_args: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    extra_link_args: Optional[List[str]] = None,
    build_ext_class: Type = None,
    verbose: bool = False,
    clean: bool = False,
) -> Tuple[str, str]:

    # Hack to remove warning about "-Wstrict-prototypes" not having effect in C++
    replaced_flags_backup = copy.deepcopy(distutils.sysconfig._config_vars)
    _clean_build_flags(distutils.sysconfig._config_vars)

    include_dirs = include_dirs or []
    library_dirs = library_dirs or []
    libraries = libraries or []
    extra_compile_args = extra_compile_args or []
    extra_link_args = extra_link_args or []

    # Build extension module
    py_extension = setuptools.Extension(
        name,
        sources,
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True), *include_dirs],
        library_dirs=[*library_dirs],
        libraries=[*libraries],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    setuptools_args = dict(
        name=name,
        ext_modules=[py_extension],
        script_args=[
            "build_ext",
            # "--parallel={}".format(gt_config.build_settings["parallel_jobs"]),
            "--build-temp={}".format(build_path),
            "--build-lib={}".format(build_path),
            "--force",
        ],
    )
    if build_ext_class is not None:
        setuptools_args["cmdclass"] = {"build_ext": build_ext_class}

    if verbose:
        setuptools_args["script_args"].append("-v")
        setuptools.setup(**setuptools_args)
    else:
        setuptools_args["script_args"].append("-q")
        io_out, io_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(io_out), contextlib.redirect_stderr(io_err):
            setuptools.setup(**setuptools_args)

    # Copy extension in target path
    module_name = py_extension._full_name
    file_path = py_extension._file_name
    src_path = os.path.join(build_path, file_path)
    dest_path = os.path.join(target_path, os.path.basename(file_path))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    distutils.file_util.copy_file(src_path, dest_path, verbose=verbose)

    # Final cleaning
    if clean:
        shutil.rmtree(build_path)

    # Restore original distutils flag config to not break functionality with "-Wstrict-prototypes"-hack for other
    # tools using distutils.
    for key, value in replaced_flags_backup.items():
        distutils.sysconfig._config_vars[key] = value

    return module_name, dest_path


# The following tells mypy to accept unpacking kwargs
@overload
def build_pybind_cuda_ext(
    name: str,
    sources: list,
    build_path: str,
    target_path: str,
    **kwargs: str,
) -> Tuple[str, str]:
    pass


def build_pybind_cuda_ext(
    name: str,
    sources: list,
    build_path: str,
    target_path: str,
    *,
    include_dirs: Optional[List[str]] = None,
    library_dirs: Optional[List[str]] = None,
    libraries: Optional[List[str]] = None,
    extra_compile_args: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    extra_link_args: Optional[List[str]] = None,
    verbose: bool = False,
    clean: bool = False,
) -> Tuple[str, str]:

    include_dirs = include_dirs or []
    include_dirs = [*include_dirs, gt_config.build_settings["cuda_include_path"]]
    library_dirs = library_dirs or []
    library_dirs = [*library_dirs, gt_config.build_settings["cuda_library_path"]]
    libraries = libraries or []
    libraries = [*libraries, "cudart"]
    extra_compile_args = extra_compile_args or []

    return build_pybind_ext(
        name,
        sources,
        build_path,
        target_path,
        verbose=verbose,
        clean=clean,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        build_ext_class=CUDABuildExtension,
    )


def _clean_build_flags(config_vars: Dict[str, str]) -> None:
    for key, value in config_vars.items():
        if type(value) == str:
            value = " " + value + " "
            for s in value.split(" "):
                if (
                    s in ["-Wstrict-prototypes", "-DNDEBUG", "-pg"]
                    or s.startswith("-O")
                    or s.startswith("-g")
                ):
                    value = value.replace(" " + s + " ", " ")
            config_vars[key] = " ".join(value.split())


class CUDABuildExtension(build_ext, object):
    # Refs:
    #   - https://github.com/pytorch/pytorch/torch/utils/cpp_extension.py
    #   - https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
    #
    def build_extensions(self) -> None:
        # Register .cu  source extensions
        self.compiler.src_extensions.append(".cu")

        # Save references to the original methods
        original_compile = self.compiler._compile

        def nvcc_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            original_compiler_so = self.compiler.compiler_so
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                if os.path.splitext(src)[-1] == ".cu":
                    nvcc_exec = os.path.join(gt_config.build_settings["cuda_bin_path"], "nvcc")
                    self.compiler.set_executable("compiler_so", [nvcc_exec])
                    if isinstance(cflags, dict):
                        cflags = cflags["nvcc"]
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler_so)

        self.compiler._compile = nvcc_compile
        build_ext.build_extensions(self)
        self.compiler._compile = original_compile
