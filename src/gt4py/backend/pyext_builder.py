# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
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
import io
import os
import shutil

import setuptools
from setuptools.command.build_ext import build_ext
import distutils.sysconfig
import pybind11

from gt4py import config as gt_config


def clean_build_flags(config_vars):
    for key, value in distutils.sysconfig._config_vars.items():
        if type(value) == str:
            value = " " + value + " "
            for s in value.split(" "):
                if (
                    s in ["-Wstrict-prototypes", "-DNDEBUG", "-pg"]
                    or s.startswith("-O")
                    or s.startswith("-g")
                ):
                    value = value.replace(" " + s + " ", " ")
            distutils.sysconfig._config_vars[key] = " ".join(value.split())


def build_pybind_ext(
    name: str,
    sources: list,
    build_path: str,
    target_path: str,
    *,
    include_dirs=None,
    library_dirs=None,
    libraries=None,
    extra_compile_args=None,
    extra_link_args=None,
    build_ext_class=None,
    verbose=False,
    clean=False,
):

    # Hack to remove warning about "-Wstrict-prototypes" not having effect in C++
    replaced_flags_backup = copy.deepcopy(distutils.sysconfig._config_vars)
    clean_build_flags(distutils.sysconfig._config_vars)

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


def build_gtcpu_ext(
    name: str,
    sources: list,
    build_path: str,
    target_path: str,
    *,
    verbose: bool = False,
    clean: bool = False,
    debug_mode: bool = False,
    add_profile_info: bool = False,
    extra_include_dirs: list = None,
):
    include_dirs = [gt_config.build_settings["boost_include_path"]]
    if extra_include_dirs:
        include_dirs.extend(extra_include_dirs)
    extra_compile_args_from_config = gt_config.build_settings["extra_compile_args"]
    if isinstance(extra_compile_args_from_config, dict):
        extra_compile_args_from_config = extra_compile_args_from_config["cxx"]

    extra_compile_args = [
        "-ftemplate-depth=2500",
        "-std=c++14",
        "-fvisibility=hidden",
        "-fopenmp",
        "-isystem{}".format(gt_config.build_settings["gt_include_path"]),
        "-isystem{}".format(gt_config.build_settings["boost_include_path"]),
        *extra_compile_args_from_config,
    ]
    extra_link_args = ["-fopenmp", *gt_config.build_settings["extra_link_args"]]

    if debug_mode:
        debug_flags = ["-O0", "-ggdb"]
        extra_compile_args.extend(debug_flags)
        extra_link_args.extend(debug_flags)
    else:
        release_flags = ["-O3", "-DNDEBUG"]
        extra_compile_args.extend(release_flags)
        extra_link_args.extend(release_flags)

    if add_profile_info:
        profile_flags = ["-pg"]
        extra_compile_args.extend(profile_flags)
        extra_link_args.extend(profile_flags)

    return build_pybind_ext(
        name,
        sources,
        build_path,
        target_path,
        verbose=verbose,
        clean=clean,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


class CUDABuildExtension(build_ext, object):
    # Refs:
    #   - https://github.com/pytorch/pytorch/torch/utils/cpp_extension.py
    #   - https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
    #
    def build_extensions(self):
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


def build_gtcuda_ext(
    name: str,
    sources: list,
    build_path: str,
    target_path: str,
    *,
    verbose: bool = False,
    clean: bool = False,
    debug_mode: bool = False,
    add_profile_info: bool = False,
    extra_include_dirs: list = None,
):
    include_dirs = [
        gt_config.build_settings["boost_include_path"],
        gt_config.build_settings["cuda_include_path"],
    ]
    if extra_include_dirs:
        include_dirs.extend(extra_include_dirs)

    library_dirs = [gt_config.build_settings["cuda_library_path"]]
    libraries = ["cudart"]

    extra_compile_args_from_config = gt_config.build_settings["extra_compile_args"]
    if isinstance(extra_compile_args_from_config, dict):
        cxx_extra_compile_args_from_config = extra_compile_args_from_config["cxx"]
        nvcc_extra_compile_args_from_config = extra_compile_args_from_config["nvcc"]
    else:
        cxx_extra_compile_args_from_config = extra_compile_args_from_config
        nvcc_extra_compile_args_from_config = extra_compile_args_from_config

    extra_compile_args = {
        "cxx": [
            "-std=c++14",
            "-fvisibility=hidden",
            "-fopenmp",
            "-isystem{}".format(gt_config.build_settings["gt_include_path"]),
            "-isystem{}".format(gt_config.build_settings["boost_include_path"]),
            "-DBOOST_PP_VARIADICS",
            "-fPIC",
            *cxx_extra_compile_args_from_config,
        ],
        "nvcc": [
            "-std=c++14",
            "-isystem={}".format(gt_config.build_settings["gt_include_path"]),
            "-isystem={}".format(gt_config.build_settings["boost_include_path"]),
            "-DBOOST_PP_VARIADICS",
            "-DBOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL",
            "-DBOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE",
            "--expt-relaxed-constexpr",
            "--compiler-options",
            "-fPIC",
            "--compiler-options",
            "-fvisibility=hidden",
            *nvcc_extra_compile_args_from_config,
        ],
    }

    extra_link_args = [*gt_config.build_settings["extra_link_args"]]

    if debug_mode:
        debug_flags = ["-O0", "-ggdb"]
        extra_compile_args["cxx"].extend(debug_flags)
        extra_compile_args["nvcc"].extend(debug_flags)
        extra_link_args.extend(debug_flags)
    else:
        release_flags = ["-O3", "-DNDEBUG"]
        extra_compile_args["cxx"].extend(release_flags)
        extra_compile_args["nvcc"].extend(release_flags)
        extra_link_args.extend(release_flags)

    if add_profile_info:
        profile_flags = ["-pg"]
        extra_compile_args["cxx"].extend(profile_flags)
        extra_compile_args["nvcc"].extend(profile_flags)
        extra_link_args.extend(profile_flags)

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
