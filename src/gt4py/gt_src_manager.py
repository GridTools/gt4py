# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""Utils for the installation and removal of GridTools C++ sources."""

import argparse
import os
import subprocess

import gt4py.config as gt_config


_DEFAULT_GRIDTOOLS_VERSION = 1
# TODO: GT2 release with CUDA SID adapter
_GRIDTOOLS_GIT_BRANCHES = {1: "release_v1.1", 2: "master"}
_GRIDTOOLS_INCLUDE_PATHS = {1: gt_config.GT_INCLUDE_PATH, 2: gt_config.GT2_INCLUDE_PATH}
_GRIDTOOLS_REPO_DIRNAMES = {1: gt_config.GT_REPO_DIRNAME, 2: gt_config.GT2_REPO_DIRNAME}


def install_gt_sources(major_version: int = _DEFAULT_GRIDTOOLS_VERSION) -> bool:
    assert major_version in _GRIDTOOLS_GIT_BRANCHES

    is_ok = has_gt_sources(major_version)
    if not is_ok:
        GIT_BRANCH = _GRIDTOOLS_GIT_BRANCHES[major_version]
        GIT_REPO = "https://github.com/GridTools/gridtools.git"

        install_path = os.path.dirname(__file__)
        target_path = os.path.abspath(
            os.path.join(install_path, "_external_src", _GRIDTOOLS_REPO_DIRNAMES[major_version])
        )
        if not os.path.exists(target_path):
            git_cmd = f"git clone --depth 1 -b {GIT_BRANCH} {GIT_REPO} {target_path}"
            print(f"Getting GridTools C++ sources...\n$ {git_cmd}")
            subprocess.check_call(git_cmd.split(), stderr=subprocess.STDOUT)

        is_ok = has_gt_sources(major_version)
        if is_ok:
            print("Success!!")
        else:
            print(
                f"\nOooops! GridTools sources have not been installed!\n"
                f"Install them manually in '{install_path}/_external_src/'\n\n"
                f"\tExample: git clone --depth 1 -b {GIT_BRANCH} {GIT_REPO} {target_path}\n"
            )

    return is_ok


def remove_gt_sources(major_version: int = _DEFAULT_GRIDTOOLS_VERSION) -> bool:
    assert major_version in _GRIDTOOLS_REPO_DIRNAMES

    install_path = os.path.dirname(__file__)
    target_path = os.path.abspath(
        os.path.join(install_path, "_external_src", _GRIDTOOLS_REPO_DIRNAMES[major_version])
    )

    is_ok = not os.path.exists(target_path)
    if not is_ok:
        rm_cmd = f"rm -Rf {target_path}"
        print(f"Deleting sources...\n$ {rm_cmd}")
        subprocess.run(rm_cmd.split())

        is_ok = not os.path.exists(target_path)
        if is_ok:
            print("Success!!")
        else:
            print(
                f"\nOooops! Something went wrong. GridTools sources may have not been removed!\n"
                f"Remove them manually from '{install_path}/_external_src/'\n\n"
                f"\tExample: rm -Rf {target_path}"
            )

    return is_ok


def has_gt_sources(major_version: int = _DEFAULT_GRIDTOOLS_VERSION) -> bool:
    assert major_version in _GRIDTOOLS_INCLUDE_PATHS
    return os.path.isfile(
        os.path.join(_GRIDTOOLS_INCLUDE_PATHS[major_version], "gridtools", "common", "defs.hpp")
    )


def _print_status(major_version: int = _DEFAULT_GRIDTOOLS_VERSION) -> None:
    if has_gt_sources(major_version):
        print("\nGridTools sources are installed\n")
    else:
        print("\nGridTools are missing (GT compiled backends will not work)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage GridTools C++ code sources")
    parser.add_argument("command", choices=["install", "remove", "status"])
    parser.add_argument(
        "-m",
        "--major-version",
        choices=list(_GRIDTOOLS_GIT_BRANCHES.keys()),
        type=int,
        default=_DEFAULT_GRIDTOOLS_VERSION,
        help=f"major GridTools version for the installation (default: {_DEFAULT_GRIDTOOLS_VERSION})",
    )
    args = parser.parse_args()

    if args.command == "install":
        install_gt_sources(args.major_version)
        _print_status(args.major_version)
    elif args.command == "remove":
        remove_gt_sources(args.major_version)
        _print_status(args.major_version)
    elif args.command == "status":
        _print_status(args.major_version)
