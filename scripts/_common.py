#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from typing import Final


PYTHON_VERSIONS: Final[list[str]] = pathlib.Path(".python-versions").read_text().splitlines()

REPO_ROOT: Final = pathlib.Path(__file__).parent.parent.resolve().absolute()
