# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from typing import Final


# DaCe passthrough prefixes
PASSTHROUGH_IN: Final[str] = "IN_"
PASSTHROUGH_OUT: Final[str] = "OUT_"

# StencilComputation in/out connector prefixes
CONNECTOR_IN: Final[str] = "__in_"
CONNECTOR_OUT: Final[str] = "__out_"

# Tasklet in/out connector prefixes
TASKLET_IN: Final[str] = "gtIN__"
TASKLET_OUT: Final[str] = "gtOUT__"
