import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import py_implementation as testee

def test_3d():
    src = np.fromfunction(lambda i, j, k : i + j + k, (3, 4, 5), dtype=np.double)
    dst = np.zeros_like(src)
    testee.copy_from_3D(src, dst)
    assert np.all(dst == src)

def test_3d_with_unit_stride():
    src = np.fromfunction(lambda i, j, k : i + j + k, (3, 4, 5), dtype=np.double)
    dst = np.zeros_like(src)
    testee.copy_from_3D_with_unit_stride(src, dst)
    assert np.all(dst == src)

def test_1d():
    shape = (3, 4, 5)
    src = np.arange(shape[0], dtype=np.double)
    dst = np.zeros(shape, dtype=np.double)
    testee.copy_from_1D(src, dst)
    expected = np.fromfunction(lambda i, j, k : i, shape, dtype=np.double)
    assert np.all(dst == expected)

def test_scalar():
    dst = np.zeros((3, 4, 5), dtype=np.double)
    testee.copy_from_scalar(42., dst)
    assert np.all(dst == 42.)

def test_cuda_sid():
    class Mock:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        @property
        def __cuda_array_interface__(self):
            return self.kwargs

    mock = Mock(
        shape=(3, 4, 5),
        typestr="|f8",
        data=(0xDEADBEAF, True),
        version=2,
        strides=(8, 3 * 8, 3 * 4 * 8),
        descr=[("", "|f8")],
        mask=None)
    testee.check_cuda_sid(mock, 0xDEADBEAF, (1, 3, 3 * 4), (3, 4, 5))
    mock = Mock(
        shape=(3, 4, 5),
        typestr="|f8",
        data=(0xDEADBEAF, True),
        version=2)
    testee.check_cuda_sid(mock, 0xDEADBEAF, (4 * 5, 5, 1), (3, 4, 5))

test_3d()
test_3d_with_unit_stride()
test_1d()
test_scalar()
test_cuda_sid()
