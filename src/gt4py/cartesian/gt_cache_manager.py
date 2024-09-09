# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for cleaning and querying the internal GT4Py cache for generated code."""

import argparse
import os
import pathlib
import shutil
from typing import List, Optional, Sequence

from gt4py.cartesian import config as gt_config


def _get_root() -> str:
    result = gt_config.cache_settings["root_path"]
    assert isinstance(result, str)
    return result


def _get_cache_name() -> str:
    result = gt_config.cache_settings["dir_name"]
    assert isinstance(result, str)
    return result


def find_caches(root: Optional[str] = None, cache_name: Optional[str] = None) -> List[pathlib.Path]:
    root_path = pathlib.Path(root or _get_root())
    cache_name = cache_name or _get_cache_name()

    result = []
    for dirpath, dirnames, _files in os.walk(root_path, topdown=True, followlinks=False):
        for d in dirnames:
            if d == cache_name:
                cache_dir = pathlib.Path(dirpath) / d
                result.append(cache_dir)

    return result


def clean_caches(caches: Sequence[pathlib.Path], *, verbose: bool = False) -> None:
    for c in caches:
        if c.exists():
            try:
                canonical_path = c.resolve(strict=True)
                if verbose:
                    print(f"\t{canonical_path} [tree]")
                shutil.rmtree(canonical_path, ignore_errors=False)
            except OSError as e:
                print(f"Error: {c} : {e.strerror}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage GT4Py cache folders")
    parser.add_argument("command", choices=["clean", "status"])
    parser.add_argument("root", nargs="*", default=[_get_root()])
    args = parser.parse_args()

    caches = [cache for root in args.root for cache in find_caches(root)]
    num_matches = len(caches)

    if args.command == "clean":
        print(f"\nCleaning cache folders: ({num_matches} found)\n")
        clean_caches(caches, verbose=True)

    elif args.command == "status":
        print(f"\nCache folder name: '{_get_cache_name()}'")
        print("\nRoot paths:")
        print(f"\tconfig = {_get_root()})")
        print(f"\targs = {args.root})")
        caches_list = "\n\t".join(str(c) for c in caches)
        print(f"\nFound {num_matches} matches{':' if num_matches > 0 else ''}\n\t{caches_list}\n")

    else:
        raise AssertionError(f"command={args.command}")
