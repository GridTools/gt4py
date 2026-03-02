# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import argparse

parser = argparse.ArgumentParser(description="Shallow Water Model")
parser.add_argument("--M", type=int, default=16, help="Number of points in the x direction")
parser.add_argument("--N", type=int, default=16, help="Number of points in the y direction")
parser.add_argument("--L_OUT", type=bool, default=True, help="a boolean for L_OUT")
parser.add_argument("--ITMAX", type=int, default=4000, help="Number of iterations")
parser.add_argument("--VAL_DEEP", type=bool, default=True, help="Do deep validation")
# parser.add_argument('--backend', type=str, default='gt:cpu_ifirst', help='Backend to use: gt:cpu_ifirst, gt:cpu_kfirst, numpy, cuda, gt:gpu')


args = parser.parse_args()

# Initialize model parameters
backend = args.backend
M = args.M
N = args.N
M_LEN = M + 1
N_LEN = N + 1
L_OUT = args.L_OUT
VAL = True
VAL_DEEP = False  # args.VAL_DEEP
VIS = False
VIS_DT = 100

ITMAX = args.ITMAX
dt = 90.0
dt = dt
dx = 100000.0
dy = 100000.0
fsdx = 4.0 / (dx)
fsdy = 4.0 / (dy)
a = 1000000.0
alpha = 0.001
