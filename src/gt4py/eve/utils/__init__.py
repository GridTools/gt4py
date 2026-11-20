# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""General utility functions. Some functionalities are directly imported from dependencies."""

from __future__ import annotations

from boltons.iterutils import (
    flatten as flatten,
    flatten_iter as flatten_iter,
    is_collection as is_collection,
)
from boltons.strutils import (
    a10n as a10n,
    asciify as asciify,
    format_int_list as format_int_list,
    iter_splitlines as iter_splitlines,
    parse_int_list as parse_int_list,
    slugify as slugify,
    unwrap_text as unwrap_text,
)

from .caching import lru_cache as lru_cache, optional_lru_cache as optional_lru_cache
from .classes import (
    CustomDefaultDictBase as CustomDefaultDictBase,
    CustomMapping as CustomMapping,
    FrozenNamespace as FrozenNamespace,
    IndexerCallable as IndexerCallable,
    Namespace as Namespace,
    SequentialIDGenerator as SequentialIDGenerator,
    get_fully_qualified_name as get_fully_qualified_name,
    is_noninstantiable as is_noninstantiable,
    noninstantiable as noninstantiable,
    register_subclasses as register_subclasses,
)
from .dispatching import (
    merge_dispatchers as merge_dispatchers,
    singledispatcher as singledispatcher,
)
from .functional import (
    attrchecker as attrchecker,
    attrgetter_ as attrgetter_,
    fluid_partial as fluid_partial,
    getitem_ as getitem_,
    isinstancechecker as isinstancechecker,
    itemgetter_ as itemgetter_,
    with_fluid_partial as with_fluid_partial,
)
from .hashing import (
    EqualityBy as EqualityBy,
    HashableBy as HashableBy,
    cached_hash as cached_hash,
    content_hash as content_hash,
    ddiff as ddiff,
    dhash as dhash,
    hashable_by as hashable_by,
    hashable_by_id as hashable_by_id,
    pprint_ddiff as pprint_ddiff,
)
from .iterators import (
    K as K,
    P as P,
    S as S,
    T as T,
    XIterable as XIterable,
    as_xiter as as_xiter,
    first as first,
    xenumerate as xenumerate,
    xiter as xiter,
)
from .strings import AnyWordsIterable as AnyWordsIterable, CaseStyleConverter as CaseStyleConverter


try:
    # For performance reasons, try to use cytoolz when possible (using cython)
    import cytoolz as toolz
except ModuleNotFoundError:
    # Fall back to pure Python toolz
    import toolz  # noqa: F401 [unused-import]  # re-exported
