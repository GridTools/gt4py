# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
from typing import Final, Optional


_DATAFILE_NAME: Final = "gt4py.json"


class BuildStatus(enum.IntEnum):
    INITIALIZED = enum.auto()
    CONFIGURED = enum.auto()
    COMPILED = enum.auto()


@dataclasses.dataclass(frozen=True)
class BuildData:
    """
    Tracks build status and results of GT4Py program compilations.

    Build system projects should keep this information up to date for each build,
    stored in a JSON file in the build folder at all times.

    The `module` property should only ever be used for compilation results which can
    directly be imported from python and contain a callable attribute named
    `entry_point_name` that represents a GT4Py program.
    """

    status: BuildStatus
    module: pathlib.Path
    entry_point_name: str

    def to_json(self) -> dict[str, str]:
        return {
            "status": self.status.name,
            "module": str(self.module),
            "entry_point_name": self.entry_point_name,
        }

    @classmethod
    def _status_to_json(cls, status: BuildStatus) -> str:
        return status.name

    @classmethod
    def _status_from_json(cls, status: str) -> BuildStatus:
        return getattr(BuildStatus, status)

    @classmethod
    def _module_from_json(cls, module: str) -> pathlib.Path:
        return pathlib.Path(module)

    @classmethod
    def from_json(cls, json_dict: dict[str, str]) -> BuildData:
        return cls(
            status=getattr(BuildStatus, json_dict["status"]),
            module=pathlib.Path(json_dict["module"]),
            entry_point_name=json_dict["entry_point_name"],
        )


def contains_data(path: pathlib.Path) -> bool:
    return (path / _DATAFILE_NAME).exists()


def read_data(path: pathlib.Path) -> Optional[BuildData]:
    try:
        return BuildData.from_json(json.loads((path / _DATAFILE_NAME).read_text()))
    except FileNotFoundError:
        return None


def write_data(data: BuildData, path: pathlib.Path) -> None:
    (path / _DATAFILE_NAME).write_text(json.dumps(data.to_json()))


def update_status(new_status: BuildStatus, path: pathlib.Path) -> None:
    old_data = read_data(path)
    assert old_data
    write_data(
        BuildData(
            status=new_status, module=old_data.module, entry_point_name=old_data.entry_point_name
        ),
        path,
    )
