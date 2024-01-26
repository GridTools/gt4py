from gt4py.next.new_type_system import inference
from gt4py.next.new_type_system import types
import numpy as np


def test_annotation_bool():
    result = inference.inferrer.from_annotation(bool)
    assert isinstance(result, types.IntegerType)
    assert result.width == 1
    assert result.signed == False


def test_annotation_int():
    result = inference.inferrer.from_annotation(int)
    assert isinstance(result, types.IntegerType)
    assert result.width == 64 or result.width == 32
    assert result.signed == True


def test_annotation_float():
    result = inference.inferrer.from_annotation(float)
    assert isinstance(result, types.FloatType)
    assert result.width == 64


def test_annotation_int8():
    result = inference.inferrer.from_annotation(np.int8)
    assert isinstance(result, types.IntegerType)
    assert result.width == 8
    assert result.signed == True


def test_annotation_int16():
    result = inference.inferrer.from_annotation(np.int16)
    assert isinstance(result, types.IntegerType)
    assert result.width == 16
    assert result.signed == True


def test_annotation_int32():
    result = inference.inferrer.from_annotation(np.int32)
    assert isinstance(result, types.IntegerType)
    assert result.width == 32
    assert result.signed == True


def test_annotation_int64():
    result = inference.inferrer.from_annotation(np.int64)
    assert isinstance(result, types.IntegerType)
    assert result.width == 64
    assert result.signed == True


def test_annotation_uint8():
    result = inference.inferrer.from_annotation(np.uint8)
    assert isinstance(result, types.IntegerType)
    assert result.width == 8
    assert result.signed == False


def test_annotation_uint16():
    result = inference.inferrer.from_annotation(np.uint16)
    assert isinstance(result, types.IntegerType)
    assert result.width == 16
    assert result.signed == False


def test_annotation_uint32():
    result = inference.inferrer.from_annotation(np.uint32)
    assert isinstance(result, types.IntegerType)
    assert result.width == 32
    assert result.signed == False


def test_annotation_uint64():
    result = inference.inferrer.from_annotation(np.uint64)
    assert isinstance(result, types.IntegerType)
    assert result.width == 64
    assert result.signed == False


def test_annotation_float32():
    result = inference.inferrer.from_annotation(np.float32)
    assert isinstance(result, types.FloatType)
    assert result.width == 32


def test_annotation_float64():
    result = inference.inferrer.from_annotation(np.float64)
    assert isinstance(result, types.FloatType)
    assert result.width == 64


def test_annotation_tuple():
    result = inference.inferrer.from_annotation(tuple[np.float32, np.float64])
    assert isinstance(result, types.TupleType)
    assert len(result.elements) == 2
    a, b = result.elements
    assert isinstance(a, types.FloatType)
    assert a.width == 32
    assert isinstance(b, types.FloatType)
    assert b.width == 64


def test_instance_bool():
    result = inference.inferrer.from_instance(False)
    assert isinstance(result, types.IntegerType)
    assert result.width == 1
    assert result.signed == False


def test_instance_int():
    result = inference.inferrer.from_instance(3)
    assert isinstance(result, types.IntegerType)
    assert result.width == 64 or result.width == 32
    assert result.signed == True


def test_instance_float():
    result = inference.inferrer.from_instance(3.14)
    assert isinstance(result, types.FloatType)
    assert result.width == 64


def test_instance_int8():
    result = inference.inferrer.from_instance(np.int8(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 8
    assert result.signed == True


def test_instance_int16():
    result = inference.inferrer.from_instance(np.int16(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 16
    assert result.signed == True


def test_instance_int32():
    result = inference.inferrer.from_instance(np.int32(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 32
    assert result.signed == True


def test_instance_int64():
    result = inference.inferrer.from_instance(np.int64(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 64
    assert result.signed == True


def test_instance_uint8():
    result = inference.inferrer.from_instance(np.uint8(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 8
    assert result.signed == False


def test_instance_uint16():
    result = inference.inferrer.from_instance(np.uint16(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 16
    assert result.signed == False


def test_instance_uint32():
    result = inference.inferrer.from_instance(np.uint32(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 32
    assert result.signed == False


def test_instance_uint64():
    result = inference.inferrer.from_instance(np.uint64(3))
    assert isinstance(result, types.IntegerType)
    assert result.width == 64
    assert result.signed == False


def test_instance_float32():
    result = inference.inferrer.from_instance(np.float32(3.14))
    assert isinstance(result, types.FloatType)
    assert result.width == 32


def test_instance_float64():
    result = inference.inferrer.from_instance(np.float64(3.14))
    assert isinstance(result, types.FloatType)
    assert result.width == 64


def test_instance_tuple():
    result = inference.inferrer.from_instance((np.float32(3.14), np.float64(3.14)))
    assert isinstance(result, types.TupleType)
    assert len(result.elements) == 2
    a, b = result.elements
    assert isinstance(a, types.FloatType)
    assert a.width == 32
    assert isinstance(b, types.FloatType)
    assert b.width == 64
