# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.storage.cartesian.layout import LayoutInfo, layout_checker_factory, layout_maker_factory


REGISTRY: dict[str, LayoutInfo] = {}


def from_name(name: str) -> LayoutInfo:
    layout_info = REGISTRY.get(name, None)
    if layout_info is None:
        raise ValueError(f"Layout '{name} is not registered. Valid options are: {REGISTRY.keys()}.")
    return layout_info


def register(name: str, info: LayoutInfo) -> None:
    if info is None:
        REGISTRY.pop(name, None)
    else:
        assert isinstance(name, str)
        assert isinstance(info, dict)

        REGISTRY[name] = info


register(
    "dace:cpu",
    LayoutInfo(
        alignment=1,
        device="cpu",
        layout_map=layout_maker_factory((1, 2, 0)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((1, 2, 0))),
    ),
)
register(
    "dace:cpu_kfirst",
    LayoutInfo(
        alignment=1,
        device="cpu",
        layout_map=layout_maker_factory((0, 1, 2)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((0, 1, 2))),
    ),
)
register(
    "dace:cpu_KJI",
    LayoutInfo(
        alignment=1,
        device="cpu",
        layout_map=layout_maker_factory((2, 1, 0)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((2, 1, 0))),
    ),
)
register(
    "dace:gpu",
    LayoutInfo(
        alignment=32,
        device="gpu",
        layout_map=layout_maker_factory((2, 1, 0)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((2, 1, 0))),
    ),
)
register(
    "debug",
    LayoutInfo(
        alignment=1,
        device="cpu",
        layout_map=layout_maker_factory((0, 1, 2)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((0, 1, 2))),
    ),
)
register(
    "gt:cpu_ifirst",
    LayoutInfo(
        alignment=1,
        device="cpu",
        layout_map=layout_maker_factory((2, 1, 0)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((2, 1, 0))),
    ),
)
register(
    "gt:cpu_kfirst",
    LayoutInfo(
        alignment=1,
        device="cpu",
        layout_map=layout_maker_factory((0, 1, 2)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((0, 1, 2))),
    ),
)
register(
    "gt:gpu",
    LayoutInfo(
        alignment=32,
        device="gpu",
        layout_map=layout_maker_factory((2, 1, 0)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((2, 1, 0))),
    ),
)
register(
    "numpy",
    LayoutInfo(
        alignment=1,
        device="cpu",
        layout_map=layout_maker_factory((0, 1, 2)),
        is_optimal_layout=layout_checker_factory(layout_maker_factory((0, 1, 2))),
    ),
)
