# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx configuration of the `gt4py.next` documentation subsite.

Pages are MyST Markdown files. The ones with a `jupytext` header declaring the
`myst` format (like `quickstart.md`) are notebooks which get executed at
build time, so their outputs are always up to date. Any other Markdown page is
rendered as is, without executing its code blocks.
"""

import pathlib
import sys

import gt4py


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from site_theme import theme_settings  # noqa: E402 [module-import-not-at-top-of-file]


project = "GT4Py Next"
copyright = "2014-2026, ETH Zurich"  # noqa: A001 [builtin-variable-shadowing]
author = "ETH Zurich"
version = gt4py.__version_info__.base_version
release = gt4py.__version__

# 'myst_nb' already loads and configures 'myst_parser'
extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]
master_doc = "index"
exclude_patterns = [
    "_build",
    # Paired notebooks generated from the '.md' sources (e.g. by 'nox -s test_examples')
    "*.ipynb",
    "**/*.ipynb",
    ".ipynb_checkpoints",
    "**/.ipynb_checkpoints",
    # Workshop material is not part of the site (yet)
    "workshop",
]

myst_enable_extensions = ["amsmath", "dollarmath"]
myst_heading_anchors = 4

# Always execute the notebooks and fail the build if any cell fails
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 600  # seconds per cell: some cells compile generated C++ code

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

# Theme settings shared by all the sites of the GT4Py documentation
globals().update(
    theme_settings(
        __file__,
        title="GT4Py Next",
        # Links to the other sites: (title, path relative to the root of this site)
        site_links=[
            ("Home", "../index.html"),
            ("gt4py.cartesian", "../cartesian/index.html"),
        ],
    )
)


def _remove_sdfg_viewer(app, doctree, docname):  # type: ignore[no-untyped-def]
    """Drop the scripts of the DaCe SDFG viewer from the notebook outputs.

    Importing `dace` in a notebook kernel displays an HTML output which loads
    the viewer (or inlines it when offline). The pages do not use it and it
    slows down their loading.
    """
    from docutils import nodes

    for node in list(doctree.findall(nodes.raw)):
        if "sdfv" in node.astext():
            node.parent.remove(node)


def setup(app):  # type: ignore[no-untyped-def]
    app.connect("doctree-resolved", _remove_sdfg_viewer)
