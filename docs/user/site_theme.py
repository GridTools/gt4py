# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""HTML theme settings shared by the Sphinx projects of the GT4Py documentation.

The documentation is made of three independent Sphinx projects (the welcome
page and the `gt4py.cartesian` and `gt4py.next` subsites) which should look the same and
link to each other. Every `conf.py` calls `theme_settings()` and adds the
result to its own configuration.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any


DOCS_DIR = pathlib.Path(__file__).parent.resolve()
REPOSITORY = "https://github.com/GridTools/gt4py"

# GridTools/CSCS red, and a lighter variant for dark backgrounds
BRAND_COLOR = "#a60b16"
BRAND_COLOR_DARK = "#f08a8a"


def theme_settings(
    conf_file: str,
    *,
    title: str,
    site_links: list[tuple[str, str]],
) -> dict[str, Any]:
    """Create the theme related settings of a Sphinx project.

    Args:
        conf_file: Path of the `conf.py` file of the project (its `__file__`).
        title: Title of the site.
        site_links: Links to the other sites of the documentation, as
            `(title, path)` pairs with paths relative to the root of this site.

    Returns:
        Sphinx configuration values, to be added to the namespace of `conf.py`.
    """
    conf_dir = pathlib.Path(conf_file).parent.resolve()
    shared = pathlib.Path(os.path.relpath(DOCS_DIR, conf_dir))
    source_directory = conf_dir.relative_to(DOCS_DIR.parent.parent).as_posix() + "/"

    settings: dict[str, Any] = {
        "html_title": title,
        "html_logo": str(shared / "_static" / "logo.svg"),
        "html_static_path": [str(shared / "_static")],
        "html_css_files": ["css/gt4py.css"],
        "templates_path": [str(shared / "_templates")],
        "html_context": {"gt4py_site_links": site_links},
    }

    settings |= {
        "html_theme": "furo",
        "html_theme_options": {
            "light_css_variables": {
                "color-brand-primary": BRAND_COLOR,
                "color-brand-content": BRAND_COLOR,
            },
            "dark_css_variables": {
                "color-brand-primary": BRAND_COLOR_DARK,
                "color-brand-content": BRAND_COLOR_DARK,
            },
            "source_repository": f"{REPOSITORY}/",
            "source_branch": "main",
            "source_directory": source_directory,
        },
        # Default sidebar plus the links to the other sites
        "html_sidebars": {
            "**": [
                "sidebar/brand.html",
                "sidebar/search.html",
                "sidebar/scroll-start.html",
                "sidebar/navigation.html",
                "gt4py-site-links.html",
                "sidebar/scroll-end.html",
                "sidebar/variant-selector.html",
            ]
        },
    }

    return settings
