# -*- coding: utf-8 -*-
import numbers


try:
    import cupy as cp
except ImportError:
    cp = None
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .definitions import Storage, SyncState


def empty(
    shape: Sequence[int],
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=False,
        data=None,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        shape=shape,
        template=None,
    )


def empty_like(
    data: Storage,
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=False,
        data=None,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        template=data,
    )


def ones(
    shape: Sequence[int],
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=True,
        data=1,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        shape=shape,
        template=None,
    )


def ones_like(
    data: Storage,
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=True,
        data=1,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        template=data,
    )


def zeros(
    shape: Sequence[int],
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=True,
        data=0,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        shape=shape,
        template=None,
    )


def zeros_like(
    data: Storage,
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=True,
        data=0,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        template=data,
    )


def full(
    shape: Sequence[int],
    fill_value: Union[float, numbers.Number],
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=True,
        data=fill_value,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        shape=shape,
        template=None,
    )


def full_like(
    data: Storage,
    fill_value: Union[float, numbers.Number],
    dtype=np.float64,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=True,
        data=fill_value,
        defaults=defaults,
        device=device,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        template=data,
    )


def as_storage(
    data: Any = None,
    dtype: Any = None,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    device_data: Any = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
    sync_state: SyncState = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=False,
        data=data,
        defaults=defaults,
        device=device,
        device_data=device_data,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        sync_state=sync_state,
        template=data,
    )


def storage(
    data: Any,
    dtype: Any = None,
    *,
    aligned_index: Optional[Sequence[int]] = None,
    alignment_size: Optional[int] = None,
    copy: bool = True,
    defaults: Optional[str] = None,
    device: Optional[str] = None,
    device_data: Any = None,
    dims: Optional[Sequence[str]] = None,
    halo: Optional[Sequence[Union[Tuple[int, int], int]]] = None,
    layout: Optional[Sequence[int]] = None,
    managed: Optional[Union[bool, str]] = None,
    sync_state: SyncState = None,
) -> Storage:

    return Storage(
        aligned_index=aligned_index,
        alignment_size=alignment_size,
        copy=copy,
        data=data,
        defaults=defaults,
        device=device,
        device_data=device_data,
        dims=dims,
        dtype=dtype,
        halo=halo,
        layout=layout,
        managed=managed,
        sync_state=sync_state,
        template=data,
    )
