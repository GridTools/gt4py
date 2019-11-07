# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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

import inspect
import os
import subprocess

from setuptools import setup, Command


class InstallGTCommand(Command):
    """A custom command to install GridTools C++ sources from the git repository."""

    description = "Install C++ GridTools sources"
    user_options = [  # (long option, short option, description)
        ("force", "f", "Overwrite existing installation"),
        ("git-repo=", None, "URL of the GridTools git repository"),
        ("git-branch=", None, "Name of the branch to be cloned"),
    ]

    def initialize_options(self):
        """Set default values for user options."""
        # Default option values
        self.force = False
        self.git_repo = "https://github.com/GridTools/gridtools.git"
        self.git_branch = "release_v1.1"

    def finalize_options(self):
        """Post-process options."""
        self.force = bool(self.force)

    def run(self):
        """Run command."""
        try:
            import gt4py
        except ImportError:
            print("ERROR: GridTools Python package must be installed first")
            return

        target_dir = "{install_dir}/_external_src/gridtools".format(
            install_dir=os.path.dirname(inspect.getabsfile(gt4py))
        )
        git_cmd = "git clone --depth 1 -b {git_branch} {git_repo} {target_dir}".format(
            git_branch=self.git_branch, git_repo=self.git_repo, target_dir=target_dir
        )
        if os.path.exists(target_dir):
            if self.force:
                subprocess.run("rm -Rf {target_dir}".format(target_dir=target_dir).split())
            else:
                print(
                    "GridTools C++ sources are already in place. Skipping...\n$ {}".format(git_cmd)
                )
                return

        print("Cloning repository...\n$ {}".format(git_cmd))
        subprocess.check_call(git_cmd.split(), stderr=subprocess.STDOUT)


class RemoveGTCommand(Command):
    """A custom command to remove GridTools C++ sources."""

    description = "Remove C++ GridTools sources"
    user_options = []  # (long option, short option, description)

    def initialize_options(self):
        """Set default values for user options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""

        try:
            import gt4py
        except ImportError:
            return

        target_dir = "{install_dir}/_external_src/gridtools".format(
            install_dir=os.path.dirname(inspect.getabsfile(gt4py))
        )
        rm_cmd = "rm -Rf {target_dir}".format(target_dir=target_dir)
        print("Deleting sources...\n$ {}".format(rm_cmd))
        subprocess.run(rm_cmd.split())


if __name__ == "__main__":
    setup(
        cmdclass={"install_gt_sources": InstallGTCommand, "remove_gt_sources": RemoveGTCommand},
        use_scm_version=True,
    )
