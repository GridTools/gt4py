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


try:
    from scipy.special import gamma as gamma_
except ImportError:
    import math

    # If scipy is not available, emulate gamma function using math.gamma
    gamma_ = np.vectorize(math.gamma)
    gamma_.types = ["f->f", "d->d", "F->F", "D->D"]


positive: np.ufunc = np.positive
negative: np.ufunc = np.negative
logical_not: np.ufunc = np.logical_not
add: np.ufunc = np.add
subtract: np.ufunc = np.subtract
multiply: np.ufunc = np.multiply
true_divide: np.ufunc = np.true_divide
greater: np.ufunc = np.greater
less: np.ufunc = np.less
greater_equal: np.ufunc = np.greater_equal
less_equal: np.ufunc = np.less_equal
equal: np.ufunc = np.equal
not_equal: np.ufunc = np.not_equal
logical_and: np.ufunc = np.logical_and
logical_or: np.ufunc = np.logical_or
abs: np.ufunc = np.abs  # noqa: A001 [builtin-variable-shadowing]
minimum: np.ufunc = np.minimum
maximum: np.ufunc = np.maximum
remainder: np.ufunc = np.remainder
sin: np.ufunc = np.sin
cos: np.ufunc = np.cos
tan: np.ufunc = np.tan
arcsin: np.ufunc = np.arcsin
arccos: np.ufunc = np.arccos
arctan: np.ufunc = np.arctan
sinh: np.ufunc = np.sinh
cosh: np.ufunc = np.cosh
tanh: np.ufunc = np.tanh
arcsinh: np.ufunc = np.arcsinh
arccosh: np.ufunc = np.arccosh
arctanh: np.ufunc = np.arctanh
sqrt: np.ufunc = np.sqrt
power: np.ufunc = np.power
exp: np.ufunc = np.exp
log: np.ufunc = np.log
log10: np.ufunc = np.log10
gamma: np.ufunc = gamma_
cbrt: np.ufunc = np.cbrt
isfinite: np.ufunc = np.isfinite
isinf: np.ufunc = np.isinf
isnan: np.ufunc = np.isnan
floor: np.ufunc = np.floor
ceil: np.ufunc = np.ceil
trunc: np.ufunc = np.trunc
