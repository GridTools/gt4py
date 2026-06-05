#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Wrapper for running the developments 'scripts' CLI within a valid venv by using a 'uv' shebang."""

from __future__ import annotations

import pathlib
import sys


if __name__ == "__main__":
    try:
        import scripts  # Import from the ./scripts folder when running as a script

        assert len(scripts.__path__) == 1 and scripts.__path__[0] == str(
            pathlib.Path(__file__).parent.resolve().absolute() / "scripts"
        ), (
            "The 'scripts' package path does not match the expected path. "
            "Please check the structure of the repository."
        )
    except ImportError as e:
        print(
            f"ERROR: '{e.name}' package cannot be imported!!\n"
            "Make sure 'uv' is installed in your system and run directly this script "
            "as an executable file, to let 'uv' create a temporary venv with all the "
            "required dependencies.\n",
            file=sys.stderr,
        )
        sys.exit(127)

    import scripts.__main__
