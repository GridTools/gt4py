# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import numpy as np


def math_builtin_test_data() -> list[tuple[str, tuple[list[int | float], ...]]]:
    return [
        # FIXME(ben): what about pow?
        # FIXME(ben): dataset is missing invalid ranges (mostly nan outputs)
        # FIXME(ben): we're not properly testing different datatypes
        # builtin name, tuple of arguments
        ("abs", ([-1, 1, -1.0, 1.0, 0, -0, 0.0, -0.0],)),
        (
            "minimum",
            (
                [2, 2.0, 2.0, 3.0, 2, 3, -2, -2.0, -2.0, -3.0, -2, -3],
                [2, 2.0, 3.0, 2.0, 3, 2, -2, -2.0, -3.0, -2.0, -3, -2],
            ),
        ),
        (
            "maximum",
            (
                [2, 2.0, 2.0, 3.0, 2, 3, -2, -2.0, -2.0, -3.0, -2, -3],
                [2, 2.0, 3.0, 2.0, 3, 2, -2, -2.0, -3.0, -2.0, -3, -2],
            ),
        ),
        (
            "fmod",
            # ([6, 6.0, -6, 6.0, 7, -7.0, 4.8, 4], [2, 2.0, 2.0, -2, 3.0, -3, 1.2, -1.2]),
            ([6.0, 7.0, 4.0], [2.0, -2.0, -3.0]),
        ),
        (
            "power",
            # ([6, 6.0, -6, 6.0, 7, -7.0, 4.8, 4], [2, 2.0, 2.0, -2, 3.0, -3, 1.2, -1.2]),
            ([2, 2.0], [2, 2.0]),
        ),
        ("sin", ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],)),
        ("cos", ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],)),
        ("tan", ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],)),
        ("arcsin", ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],)),
        ("arccos", ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],)),
        (
            "arctan",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "sinh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "cosh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "tanh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "arcsinh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        ("arccosh", ([1, 1.0, 1.2, 1.7, 2, 2.0, 100, 103.7, 1000, 1379.89],)),
        ("arctanh", ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],)),
        (
            "sqrt",
            (
                [
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.9,
                    1,
                    1.0,
                    2.3,
                    4,
                    4.0,
                    16,
                    16.0,
                    34.7,
                    100,
                    100.0,
                    1000,
                    1337.1337,
                ],
            ),
        ),
        (
            "exp",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "log",
            (
                [
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.9,
                    1,
                    1.0,
                    2.3,
                    4,
                    4.0,
                    16,
                    16.0,
                    34.7,
                    100,
                    100.0,
                    1000,
                    1337.1337,
                ],
            ),
        ),
        (
            "gamma",
            # TODO(ben): math.gamma throws when it overflows, maybe should instead yield `np.inf`?
            #  overflows very quickly, already at `173`
            ([-1002.3, -103.7, -1.2, -0.7, -0.1, 0.1, 0.7, 1.0, 1, 1.2, 100, 103.7, 170.5],),
        ),
        (
            "cbrt",
            (
                [
                    -1003.2,
                    -704.3,
                    -100.5,
                    -10.4,
                    -1.5,
                    -1.001,
                    -0.7,
                    -0.01,
                    -0.0,
                    0.0,
                    0.01,
                    0.7,
                    1.001,
                    1.5,
                    10.4,
                    100.5,
                    704.3,
                    1003.2,
                ],
            ),
        ),
        ("isfinite", ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],)),
        ("isinf", ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],)),
        (
            "isnan",
            # TODO(BenWeber42): would be good to ensure we have nans with different bit patterns
            ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],),
        ),
        ("floor", ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],)),
        ("ceil", ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],)),
        ("trunc", ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],)),
    ]
