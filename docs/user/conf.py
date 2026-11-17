# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx configuration of the GT4Py documentation welcome page.

The documentation site is made of three independent Sphinx projects: this
welcome page and the `cartesian/` and `next/` subsites, which are built into
subfolders of the output of this one (see the `docs` session in `noxfile.py`).
"""

import pathlib
import sys

import gt4py


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from site_theme import theme_settings  # noqa: E402 [module-import-not-at-top-of-file]


project = "GT4Py"
copyright = "2014-2026, ETH Zurich"  # noqa: A001 [builtin-variable-shadowing]
author = "ETH Zurich"
version = gt4py.__version_info__.base_version
release = gt4py.__version__

extensions = ["myst_parser", "sphinx_design"]
source_suffix = {".md": "markdown"}
master_doc = "index"
# The subsites are separate Sphinx projects with their own configuration
exclude_patterns = ["_build", "cartesian", "next"]

# Theme settings shared by all the sites of the GT4Py documentation
globals().update(
    theme_settings(
        __file__,
        title="GT4Py",
        # Links to the other sites: (title, path relative to the root of this site)
        site_links=[
            ("gt4py.next", "next/index.html"),
            ("gt4py.cartesian (previous generation)", "cartesian/index.html"),
        ],
    )
)
html_show_sourcelink = False
