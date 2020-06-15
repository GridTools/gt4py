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

import argparse
import os
import subprocess

from . import config as gt_config


def install_gt_sources():
    is_ok = has_gt_sources()
    if not is_ok:
        GIT_BRANCH = "release_v1.1"
        GIT_REPO = "https://github.com/GridTools/gridtools.git"

        install_path = os.path.dirname(__file__)
        target_path = os.path.abspath(os.path.join(install_path, "_external_src", "gridtools"))
        if not os.path.exists(target_path):
            git_cmd = f"git clone --depth 1 -b {GIT_BRANCH} {GIT_REPO} {target_path}"
            print(f"Getting GridTools C++ sources...\n$ {git_cmd}")
            subprocess.check_call(git_cmd.split(), stderr=subprocess.STDOUT)

        is_ok = has_gt_sources()
        if is_ok:
            print("Success!!")
        else:
            print(
                f"\nOooops! GridTools sources have not been installed!\n"
                f"Install them manually in '{install_path}/_external_src/'\n\n"
                f"\tExample: git clone --depth 1 -b {GIT_BRANCH} {GIT_REPO}\n"
            )

    return is_ok


def remove_gt_sources():
    install_path = os.path.dirname(__file__)
    target_path = os.path.abspath(os.path.join(install_path, "_external_src", "gridtools"))

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


def has_gt_sources():
    return os.path.isfile(
        os.path.join(gt_config.GT_INCLUDE_PATH, "gridtools", "common", "defs.hpp")
    )


def _print_status():
    if has_gt_sources():
        print("\nGridTools sources are installed\n")
    else:
        print("\nGridTools are missing (GT compiled backends will not work)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage GridTools C++ code sources")
    parser.add_argument("command", choices=["install", "remove", "status"])
    args = parser.parse_args()

    if args.command == "install":
        install_gt_sources()
        _print_status()
    elif args.command == "remove":
        remove_gt_sources()
        _print_status()
    elif args.command == "status":
        _print_status()
