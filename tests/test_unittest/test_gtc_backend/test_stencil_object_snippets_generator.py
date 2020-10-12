from typing import Iterator

import pytest

from gt4py.backend.gtc_backend.common import DataType
from gt4py.backend.gtc_backend.gtir import AccessKind, FieldBoundary, FieldMetadata, FieldsMetadata
from gt4py.backend.gtc_backend.stencil_object_snippet_generators import FieldInfoGenerator


@pytest.fixture
def generator() -> Iterator[FieldInfoGenerator]:
    yield FieldInfoGenerator()


def test_field_boundary(generator) -> None:
    field_boundary = FieldBoundary(i=(0, 1), j=(2, 3), k=(4, 5))
    assert generator.apply(field_boundary) == "Boundary((0, 1), (2, 3), (4, 5))"


def test_field_metadata(generator) -> None:
    field_metadata = FieldMetadata(
        name="a",
        access=AccessKind.READ_ONLY,
        boundary=FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
        dtype=DataType.FLOAT64,
    )
    assert generator.apply(field_metadata) == (
        "'a': FieldInfo(access=AccessKind.READ_ONLY, "
        "boundary=Boundary((0, 0), (0, 0), (0, 0)), "
        "dtype=dtype('float64'))"
    )


def test_fields_metadata(generator) -> None:
    fields_metadata = FieldsMetadata(
        metas={
            "a": FieldMetadata(
                name="a",
                access=AccessKind.READ_ONLY,
                boundary=FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
                dtype=DataType.FLOAT64,
            ),
            "b": FieldMetadata(
                name="b",
                access=AccessKind.READ_WRITE,
                boundary=FieldBoundary(i=(1, 2), j=(3, 4), k=(5, 6)),
                dtype=DataType.FLOAT64,
            ),
        }
    )
    assert generator.apply(fields_metadata) == (
        "{'a': FieldInfo(access=AccessKind.READ_ONLY, "
        "boundary=Boundary((0, 0), (0, 0), (0, 0)), "
        "dtype=dtype('float64')), "
        "'b': FieldInfo(access=AccessKind.READ_WRITE, "
        "boundary=Boundary((1, 2), (3, 4), (5, 6)), "
        "dtype=dtype('float64'))}"
    )
