# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import pathlib
import threading
from typing import Generator

from flufl import lock as flufl_lock


@contextlib.contextmanager
def refreshing_lock(
    path: pathlib.Path, lifetime: int = 10, refresh: int = 5
) -> Generator[None, None, None]:
    lock = flufl_lock.Lock(str(path), lifetime=lifetime)  # type: ignore[attr-defined] # mypy doesn't understand flufl.lock's custom export mechanism
    lock.lock()
    stop_event = threading.Event()

    def alive() -> None:
        while not stop_event.wait(refresh):
            lock.refresh(lifetime)

    alive_thread = threading.Thread(target=alive, daemon=True)
    alive_thread.start()
    yield
    stop_event.set()
    alive_thread.join()
    lock.unlock()
