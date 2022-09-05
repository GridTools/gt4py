# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
from typing import Final, Optional


_DATAFILE_NAME: Final = "gt4py.json"

OTFBuildStatus = enum.IntEnum("OTFBuildStatus", ["STARTED", "CONFIGURED", "COMPILED"], start=0)


@dataclasses.dataclass(frozen=True)
class OTFBuildData:
    status: OTFBuildStatus
    module: pathlib.Path
    entry_point_name: str

    def to_json(self) -> dict[str, str]:
        return dataclasses.asdict(self) | {"status": self.status.name, "module": str(self.module)}

    @classmethod
    def from_json(cls, data) -> OTFBuildData:
        return cls(
            **(
                data
                | {
                    "status": getattr(OTFBuildStatus, data["status"]),
                    "module": pathlib.Path(data["module"]),
                }
            )
        )


def contains_data(path: pathlib.Path) -> bool:
    return (path / _DATAFILE_NAME).exists()


def read_data(path) -> Optional[OTFBuildData]:
    if contains_data(path):
        return OTFBuildData.from_json(json.loads((path / _DATAFILE_NAME).read_text()))
    return None


def write_data(data: OTFBuildData, path: pathlib.Path) -> None:
    (path / _DATAFILE_NAME).write_text(json.dumps(data.to_json()))


def update_status(new_status: OTFBuildStatus, path: pathlib.Path) -> None:
    old_data = read_data(path)
    assert old_data
    write_data(
        OTFBuildData(
            status=new_status, module=old_data.module, entry_point_name=old_data.entry_point_name
        ),
        path,
    )
