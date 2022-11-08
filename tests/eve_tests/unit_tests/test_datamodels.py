# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

import enum
import types
import typing
from typing import Set  # noqa: F401  # imported but unused (used in exec() context)
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    ForwardRef,
    Generic,
    List,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import devtools
import factory
import pytest
import pytest_factoryboy as pytfboy

from eve import datamodels, utils


T = TypeVar("T")
U = TypeVar("U")


SAMPLE_MODEL_FACTORIES: List[factory.Factory] = []


def register_factory(
    factory_class: Optional[factory.Factory] = None,
    *,
    model_fixture: Optional[str] = None,
    collection: Optional[MutableSequence[factory.Factory]] = None,
) -> Union[factory.Factory, Callable[[factory.Factory], factory.Factory]]:
    def _decorator(factory: factory.Factory) -> factory.Factory:
        if collection is not None:
            collection.append(factory)
        return pytfboy.register(factory, model_fixture)

    return _decorator(factory_class) if factory_class is not None else _decorator


class SampleEnum(enum.Enum):
    FOO = "foo"
    BLA = "bla"


class SampleClass:
    pass


class EmptyModel(datamodels.DataModel):
    pass


@register_factory(collection=SAMPLE_MODEL_FACTORIES)
class EmptyModelFactory(factory.Factory):
    class Meta:
        model = EmptyModel


class IntModel(datamodels.DataModel):
    value: int


@register_factory(collection=SAMPLE_MODEL_FACTORIES)
class IntModelFactory(factory.Factory):
    class Meta:
        model = IntModel

    value = 1


class AnyModel(datamodels.DataModel):
    value: Any


@register_factory(collection=SAMPLE_MODEL_FACTORIES)
class AnyModelFactory(factory.Factory):
    class Meta:
        model = AnyModel

    value = ("any", "value")


class GenericModel(datamodels.GenericDataModel, Generic[T]):
    value: T


@register_factory(collection=SAMPLE_MODEL_FACTORIES)
class GenericModelFactory(factory.Factory):
    class Meta:
        model = GenericModel

    value = "generic value"


@pytest.fixture(params=SAMPLE_MODEL_FACTORIES)
def example_model_factory(request) -> datamodels.DataModel:
    return request.param


# --- Tests ---
# Test generated members
def test_datamodel_class_members(example_model_factory):
    model = example_model_factory()
    model_class = model.__class__

    assert hasattr(model_class, "__init__")
    assert hasattr(model_class, "__datamodel_fields__")
    assert isinstance(model_class.__datamodel_fields__, utils.FrozenNamespace)
    assert hasattr(model_class, "__datamodel_params__")
    assert isinstance(model_class.__datamodel_params__, utils.FrozenNamespace)
    assert hasattr(model_class, "__datamodel_root_validators__")
    assert isinstance(model_class.__datamodel_root_validators__, tuple)


def test_devtools_compatibility(example_model_factory):
    model = example_model_factory()
    model_class = model.__class__
    formatted_string = devtools.pformat(model)

    assert hasattr(model_class, "__pretty__")
    assert callable(model_class.__pretty__)
    assert f"{model_class.__name__}(" in formatted_string
    for name in model_class.__datamodel_fields__.keys():
        assert f"{name}=" in formatted_string


class TestInitialization:
    def test_init(self):
        @datamodels.datamodel
        class Model:
            value: int
            enum_value: SampleEnum
            list_value: List[int]

        model = Model(value=1, enum_value=SampleEnum.FOO, list_value=[1, 2, 3])
        assert model.value == 1
        assert model.enum_value == SampleEnum.FOO
        assert model.list_value == [1, 2, 3]

    def test_custom_init(self):
        class ModelWithCustomInit(datamodels.DataModel):
            value: float
            enum_value: SampleEnum
            list_value: List[float]

            def __init__(self, single_value: float) -> None:
                self.__auto_init__(single_value, SampleEnum.BLA, [1.0] * int(single_value))

        model = ModelWithCustomInit(3.5)
        assert model.value == 3.5
        assert model.enum_value == SampleEnum.BLA
        assert model.list_value == [1.0, 1.0, 1.0]

    def test_init_hooks(self):
        @datamodels.datamodel
        class ModelWithInitHooks:
            value: float
            STATIC_INT: ClassVar[int] = 0

            def __pre_init__(self) -> None:
                self.__class__.STATIC_INT += 1

            def __post_init__(self) -> None:
                self.value *= 10

        assert ModelWithInitHooks.STATIC_INT == 0
        model = ModelWithInitHooks(3.5)
        assert ModelWithInitHooks.STATIC_INT == 1
        assert model.value == 3.5 * 10

    def test_custom_init_and_hooks(self):
        class ModelWithCustomInitAndHooks(datamodels.DataModel):
            value: float
            STATIC_INT: ClassVar[int] = 0

            def __init__(self, str_value: str) -> None:
                self.__auto_init__(float(str_value))

            def __pre_init__(self) -> None:  # type: ignore[override]
                self.__class__.STATIC_INT += 1

            def __post_init__(self) -> None:  # type: ignore[override]
                self.value *= 10

        assert ModelWithCustomInitAndHooks.STATIC_INT == 0
        model = ModelWithCustomInitAndHooks("-5.25")
        assert ModelWithCustomInitAndHooks.STATIC_INT == 1
        assert model.value == -5.25 * 10

    def test_non_datamodels_init(self):
        @datamodels.datamodel
        class ModelWithReallyCustomInit:
            value: float
            STATIC_INT: ClassVar[int] = 0

            def __init__(self, str_value: str) -> None:
                self.value = float(str_value)

            def __pre_init__(self) -> None:
                self.__class__.STATIC_INT += 1

            def __post_init__(self) -> None:
                self.value *= 10

        assert ModelWithReallyCustomInit.STATIC_INT == 0
        model = ModelWithReallyCustomInit("-5.25")
        assert ModelWithReallyCustomInit.STATIC_INT == 0
        assert model.value == -5.25


def test_kw_only_model():
    @datamodels.datamodel
    class Model:
        int_value: int
        str_value: str

    @datamodels.datamodel(kw_only=True)
    class KwModel:
        int_value: int
        str_value: str

    @datamodels.datamodel(kw_only=False)
    class NotKwModel:
        int_value: int
        str_value: str

    model = Model(1, "foo")
    kw_model = KwModel(int_value=1, str_value="foo")
    not_kw_model = NotKwModel(1, str_value="foo")
    assert (model.int_value, model.str_value) == (kw_model.int_value, kw_model.str_value)
    assert (model.int_value, model.str_value) == (not_kw_model.int_value, not_kw_model.str_value)

    with pytest.raises(TypeError, match="takes 1 positional argument but 3 were given"):
        KwModel(1, "foo")

    with pytest.raises(TypeError, match="takes 1 positional argument but 2 positional"):
        KwModel(1, str_value="foo")

    @datamodels.datamodel
    class MixedModel:
        int_value: int
        str_value: str = datamodels.field(kw_only=True)

    assert MixedModel(int_value=3, str_value="foo") == MixedModel(3, str_value="foo")

    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        MixedModel(3, "foo")


def test_slots():
    class Model:
        as_int: int
        a_str: str

    assert "__slots__" not in Model.__dict__

    SlottedModel = datamodels.datamodel(Model, slots=True)
    model = SlottedModel(33, "foo")

    assert (model.as_int, model.a_str) == (33, "foo")
    assert "__dict__" not in dir(model)
    assert "__slots__" in SlottedModel.__dict__
    assert SlottedModel is not Model

    DataModel = datamodels.datamodel(Model, slots=False)
    model = DataModel(33, "foo")

    assert (model.as_int, model.a_str) == (33, "foo")
    assert "__dict__" in dir(model)
    assert "__slots__" not in DataModel.__dict__
    assert DataModel is Model


class TestConversion:
    def test_simple_coertion_in_decorator(self):
        @datamodels.datamodel(coerce=True)
        class CoercedModel:
            as_int: int
            a_str: str

        instance = CoercedModel(-2, "2")
        assert instance.as_int == -2
        assert instance.a_str == "2"

        instance = CoercedModel(-2, 2)
        assert instance.as_int == -2
        assert instance.a_str == "2"

        A_STR = """
            Lorem ipsum dolor sit amet, nonumy accusam suscipit et mei, ipsum saperet no nec,
            te volumus insolens nam. Verear scripserit delicatissimi cu vis, eam graeci facete in.
            Atqui inani maiorum sea ex. Vim vidit intellegam eu. Mei dico lorem eu, at per paulo
            aperiri admodum. Summo iriure consequuntur per ea, his ex amet tacimates.
        """
        instance = CoercedModel("-2", A_STR)
        assert instance.as_int == -2
        assert instance.a_str == A_STR
        assert instance.a_str is A_STR

    def test_simple_coertion_in_fields(self):
        class PartiallyCoercedModel(datamodels.DataModel):
            as_int: int = datamodels.field(converter="coerce")
            only_int: int

        instance = PartiallyCoercedModel(-2, 2)
        assert instance.as_int == -2
        assert instance.only_int == 2

        instance = PartiallyCoercedModel("-2", 2)
        assert instance.as_int == -2
        assert instance.only_int == 2

        with pytest.raises(TypeError, match="only_int"):
            PartiallyCoercedModel("-2", "2")

        class PartiallyCoercedModel(datamodels.DataModel):
            as_int: datamodels.Coerced[int]
            only_int: int

        instance = PartiallyCoercedModel(-2, 2)
        assert instance.as_int == -2
        assert instance.only_int == 2

        instance = PartiallyCoercedModel("-2", 2)
        assert instance.as_int == -2
        assert instance.only_int == 2

        with pytest.raises(TypeError, match="only_int"):
            PartiallyCoercedModel("-2", "2")

    def test_custom_coertion_in_fields(self):
        class CustomCoercedModel(datamodels.DataModel):
            as_int: int = datamodels.field(converter=int)
            only_int: int

        instance = CustomCoercedModel(-2, 2)
        assert instance.as_int == -2
        assert instance.only_int == 2

        instance = CustomCoercedModel("-2", 2)
        assert instance.as_int == -2
        assert instance.only_int == 2

        with pytest.raises(TypeError, match="only_int"):
            CustomCoercedModel("-2", "2")

        with pytest.raises(
            TypeError, match="'CustomCoercedModel.as_int' which already defines a custom converter"
        ):

            class CustomCoercedModel(datamodels.DataModel):
                as_int: datamodels.Coerced[int] = datamodels.field(converter=int)
                only_int: int

    def test_coertion_of_collections(self):
        @datamodels.datamodel(coerce=True)
        class ModelWithCollections:
            items: List[int]
            table: Dict[int, Tuple[str, str]]

        instance = ModelWithCollections([], {})
        assert instance.items == []
        assert instance.table == {}

        instance = ModelWithCollections([1, 2], {2: ("two", "zwei")})
        assert instance.items == [1, 2]
        assert instance.table == {2: ("two", "zwei")}

        instance = ModelWithCollections((-1, -2), [(2, ("two", "zwei")), (3, ("three", "drei"))])
        assert instance.items == [-1, -2]
        assert instance.table == {2: ("two", "zwei"), 3: ("three", "drei")}

        with pytest.raises(TypeError, match="items"):
            instance = ModelWithCollections(["1", "2"], {})
        with pytest.raises(TypeError, match="items"):
            instance = ModelWithCollections(("1", "2"), {})


class TestAnnotationMarkers:
    def test_coerced_marker(self):
        @datamodels.datamodel
        class CoercedModel:
            as_int: datamodels.Coerced[int]
            only_str: str

        instance = CoercedModel(-2, "2")
        assert instance.as_int == -2
        assert instance.only_str == "2"

        instance = CoercedModel("-2", "2")
        assert instance.as_int == -2
        assert instance.only_str == "2"

        with pytest.raises(TypeError, match="only_str"):
            CoercedModel("-2", 2)

    def test_unchecked_marker(self):
        @datamodels.datamodel
        class UncheckedModel:
            as_int: datamodels.Unchecked[int]
            only_int: int

        instance = UncheckedModel(-2, 2)
        assert instance.as_int == -2
        assert instance.only_int == 2

        instance = UncheckedModel(-2.5, 2)
        assert instance.as_int == -2.5
        assert instance.only_int == 2

        with pytest.raises(TypeError, match="only_int"):
            instance = UncheckedModel(-2.5, 2.5)

    def test_combined_markers(self):
        @datamodels.datamodel
        class UncheckedAndCoercedModel:
            a: datamodels.Unchecked[datamodels.Coerced[int]]
            b: datamodels.Coerced[datamodels.Unchecked[int]]

        instance = UncheckedAndCoercedModel(-2.5, "-2")
        assert instance.a == -2 == instance.b

        instance = UncheckedAndCoercedModel("-2", -2.5)
        assert instance.a == -2 == instance.b


class TestTypeValidationFactory:
    def test_no_factory(self):
        @datamodels.datamodel(type_validation_factory=None)
        class UnsafeModel:
            an_int: int
            a_float: float

        model = UnsafeModel("foo", "bar")
        assert isinstance(model, UnsafeModel)
        assert model.an_int, model.a_float == ("foo", "bar")

    def test_custom_type_validation_factory(self):
        def custom_factory(type_hint, name):
            if type_hint is int:

                def _validator(model, attrib, value):
                    if value % 2:
                        raise TypeError("FOO")

                return _validator

            else:
                return lambda model, attrib, value: None

        @datamodels.datamodel(type_validation_factory=custom_factory)
        class BooModel:
            an_int: int
            a_float: float

        model = BooModel(2, 2.0)
        assert model.an_int, model.a_float == (2, 2.0)

        model = BooModel(2, "a float")
        assert model.an_int, model.a_float == (2, "a float")

        with pytest.raises(TypeError, match="FOO"):
            model = BooModel(3, 2.0)


def test_default_values():
    @datamodels.datamodel
    class Model:
        bool_value: bool
        int_value: int = 1
        enum_value: SampleEnum = SampleEnum.FOO
        any_value: Any = datamodels.field(default="ANY")

    model = Model(False)
    with pytest.raises(TypeError, match="'bool_value'"):
        Model()

    assert model.bool_value is False
    assert model.int_value == 1
    assert model.enum_value == SampleEnum.FOO
    assert model.any_value == "ANY"

    @datamodels.datamodel
    class WrongModel:
        bool_value: bool = 1

    with pytest.raises(TypeError, match="'WrongModel.bool_value'"):
        WrongModel()


def test_default_factories():
    @datamodels.datamodel
    class Model:
        list_value: List[int] = datamodels.field(default_factory=list)
        dict_value: Dict[str, int] = datamodels.field(default_factory=lambda: {"one": 1})

    model = Model()

    assert model.list_value == []
    assert model.dict_value == {"one": 1}

    @datamodels.datamodel
    class WrongModel:
        list_value: List[int] = datamodels.field(default_factory=tuple)

    with pytest.raises(TypeError, match="'WrongModel.list_value'"):
        WrongModel()


# Test field specification
SAMPLE_TYPE_DATA: Final = [
    ("bool", [True, False], [1, "True"]),
    ("int", [1, -1], [1.0, True, "1"]),
    ("float", [1.0], [1, "1.0"]),
    ("str", ["", "one"], [1, ("one",)]),
    ("complex", [1j], [1, 1.0, "1j"]),
    ("bytes", [b"bytes", b""], ["string", ["a"]]),
    ("typing.Any", ["any"], tuple()),
    ("typing.Literal[1, 1.0, True]", [1, 1.0, True], [False]),
    ("typing.Tuple[int, str]", [(3, "three")], [(), (3, 3)]),
    ("typing.Tuple[int, ...]", [(1, 2, 3), ()], [3, (3, "three")]),
    ("typing.List[int]", ([1, 2, 3], []), (1, [1.0])),
    ("typing.Set[int]", ({1, 2, 3}, set()), (1, [1], (1,), {1: None})),
    ("typing.Dict[int, str]", ({}, {3: "three"}), ([(3, "three")], 3, "three", [])),
    ("typing.Sequence[int]", ([1, 2, 3], [], (1, 2, 3), tuple()), (1, [1.0], {1})),
    ("typing.MutableSequence[int]", ([1, 2, 3], []), ((1, 2, 3), tuple(), 1, [1.0], {1})),
    ("typing.Set[int]", ({1, 2, 3}, set()), (1, [1], (1,), {1: None})),
    ("typing.Union[int, float, str]", [1, 3.0, "one"], [[1], [], 1j]),
    ("typing.Optional[int]", [1, None], [[1], [], 1j]),
    (
        "typing.Dict[Union[int, float, str], Union[Tuple[int, Optional[float]], Set[int]]]",
        [{1: (2, 3.0)}, {1.0: (2, None)}, {"1": {1, 2}}],
        [{(1, 1.0, "1"): set()}, {1: [1]}, {"1": (1,)}],
    ),
]


@pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], SAMPLE_TYPE_DATA)
def test_field_type_hint(type_hint: str, valid_values: Sequence[Any], wrong_values: Sequence[Any]):
    context: Dict[str, Any] = {}
    exec(
        f"""
@datamodels.datamodel
class Model:
    value: {type_hint}
""",
        globals(),
        context,
    )
    Model = context["Model"]

    field_type = Model.__datamodel_fields__.value.type
    if isinstance(field_type, str):
        field_type = eval(field_type)
    assert field_type == eval(type_hint)

    for value in valid_values:
        Model(value=value)

    for value in wrong_values:
        with pytest.raises((TypeError, ValueError), match="'Model.value'"):
            Model(value=value)


def test_custom_class_type_hint():
    @datamodels.datamodel
    class Model1:
        value: SampleEnum

    Model1(value=SampleEnum.FOO)

    class SampleClassChild(SampleClass):
        pass

    class SampleClassGrandChild(SampleClassChild):
        pass

    @datamodels.datamodel
    class Model2:
        value: SampleClass

    Model2(value=SampleClass())
    Model2(value=SampleClassChild())
    Model2(value=SampleClassGrandChild())

    class AnotherClass:
        pass

    with pytest.raises(TypeError, match="value"):
        Model1(value=AnotherClass())
    with pytest.raises(TypeError, match="value"):
        Model2(value=AnotherClass())


class MyType:
    def __init__(self, value: Any) -> None:
        self.value = value

    def add(self, something) -> None:
        return self.value + something

    @classmethod
    def __type_validator__(cls) -> datamodels.FieldValidator:
        def _custom_validator(
            instance: datamodels.DataModel, attribute: datamodels.Attribute, value: Any
        ) -> None:
            if not (hasattr(value, "value") and hasattr(value, "add")):
                raise TypeError("Invalid value type for '{attribute.name}' field.")

        return _custom_validator


@datamodels.datamodel
class GlobalRecursiveModel:
    value: Optional[GlobalRecursiveModel] = None
    others: Optional[Dict[Literal["A", "B"], Union[str, GlobalRecursiveModel]]] = None


def test_deferred_class_type_hint():
    # Model defined in a module global context
    assert isinstance(GlobalRecursiveModel.__datamodel_fields__.value.type, ForwardRef)
    assert isinstance(GlobalRecursiveModel.__datamodel_fields__.others.type, ForwardRef)

    m1 = GlobalRecursiveModel()
    m2 = GlobalRecursiveModel(value=m1)
    m3 = GlobalRecursiveModel(value=m2)
    GlobalRecursiveModel(value=m2, others={"A": m3, "B": "something"})

    with pytest.raises(TypeError, match="value"):
        GlobalRecursiveModel(value="wrong_value")
    with pytest.raises(TypeError, match="others"):
        GlobalRecursiveModel(others={"A": -1})
    with pytest.raises(TypeError, match="others"):
        GlobalRecursiveModel(others={"a": "wrong"})

    assert GlobalRecursiveModel.__datamodel_fields__.value.type.__args__[0] == GlobalRecursiveModel
    assert (
        GlobalRecursiveModel.__datamodel_fields__.others.type.__args__[0].__args__[1].__args__[1]
        == GlobalRecursiveModel
    )

    # Models defined in a non-global context
    @datamodels.datamodel
    class RecursiveModel:
        int_value: int
        list_value: List[RecursiveModel]

    assert isinstance(RecursiveModel.__datamodel_fields__.list_value.type, ForwardRef)
    datamodels.update_forward_refs(RecursiveModel, {"RecursiveModel": RecursiveModel})
    assert RecursiveModel.__datamodel_fields__.list_value.type.__args__[0] == RecursiveModel

    m1 = RecursiveModel(int_value=1, list_value=[])
    m2 = RecursiveModel(int_value=1, list_value=[m1])
    RecursiveModel(int_value=1, list_value=[m1, m2])

    with pytest.raises(TypeError, match="list_value"):
        RecursiveModel(int_value=1, list_value=["wrong_value"])

    @datamodels.datamodel
    class CollectorModel:
        value1: NotYetDefinedModel1
        value2: NotYetDefinedModel2

    @datamodels.datamodel
    class NotYetDefinedModel1:
        int_value: int

    @datamodels.datamodel
    class NotYetDefinedModel2:
        int_value: int

    assert isinstance(CollectorModel.__datamodel_fields__.value1.type, ForwardRef)
    assert isinstance(CollectorModel.__datamodel_fields__.value2.type, ForwardRef)
    datamodels.update_forward_refs(
        CollectorModel,
        {"NotYetDefinedModel1": NotYetDefinedModel1, "NotYetDefinedModel2": NotYetDefinedModel2},
    )
    assert CollectorModel.__datamodel_fields__.value1.type == NotYetDefinedModel1
    assert CollectorModel.__datamodel_fields__.value2.type == NotYetDefinedModel2

    CollectorModel(value1=NotYetDefinedModel1(int_value=1), value2=NotYetDefinedModel2(int_value=2))
    with pytest.raises(TypeError, match="value2"):
        CollectorModel(value1=NotYetDefinedModel1(int_value=1), value2=2)


def test_field_redefinition():
    class Model(datamodels.DataModel):
        value: int = 0

    assert Model().value == 0

    # Redefinition with same type
    class ChildModel1(Model):
        value: int = 1

    assert ChildModel1().value == 1

    # Redefinition with different type
    class ChildModel2(Model):
        value: float = 2.2  # type: ignore  # redefining value as float on purpose

    assert ChildModel2().value == 2.2

    with pytest.raises(TypeError, match="value"):
        assert ChildModel2(value=2)


def test_class_vars():
    @datamodels.datamodel
    class Model:
        value: Any
        default_value: Any = None

        class_var: ClassVar[int] = 0

    field_names = set(Model.__datamodel_fields__.keys())
    assert field_names == {"value", "default_value"}
    assert hasattr(Model, "class_var")
    assert Model.class_var == 0


def test_missing_annotations():
    with pytest.raises(TypeError, match="other_value"):

        class Model(datamodels.DataModel):
            other_value = datamodels.field(default=None)


# Test custom validators
class ModelWithValidators(datamodels.DataModel):
    bool_value: bool = False
    int_value: int = 0
    even_int_value: int = 2
    float_value: float = 0.0
    str_value: str = ""
    extra_value: Optional[Any] = None

    @datamodels.validator("bool_value")
    def _bool_value_validator(
        self: datamodels.DataModel, attribute: datamodels.Attribute, value: Any
    ) -> None:
        assert isinstance(self, ModelWithValidators)

    @datamodels.validator("int_value")
    def _int_value_validator(
        self: datamodels.DataModel, attribute: datamodels.Attribute, value: Any
    ) -> None:
        if value < 0:
            raise ValueError(f"'{attribute.name}' must be larger or equal to 0")

    @datamodels.validator("even_int_value")
    def _even_int_value_validator(
        self: datamodels.DataModel, attribute: datamodels.Attribute, value: Any
    ) -> None:
        if value % 2:
            raise ValueError(f"'{attribute.name}' must be an even number")

    @datamodels.validator("float_value")
    def _float_value_validator(
        self: datamodels.DataModel, attribute: datamodels.Attribute, value: Any
    ) -> None:
        if value > 3.14159:
            raise ValueError(f"'{attribute.name}' must be lower or equal to 3.14159")

    @datamodels.validator("str_value")
    def _str_value_validator(
        self: datamodels.DataModel, attribute: datamodels.Attribute, value: Any
    ) -> None:
        # This kind of validation should arguably happen in a root_validator, but
        # since float_value is defined before str_value, it should have been
        # already validated at this point
        if value == str(cast(ModelWithValidators, self).float_value):
            raise ValueError(f"'{attribute.name}' must be different to 'float_value'")

    @datamodels.validator("extra_value")
    def _extra_value_validator(
        self: datamodels.DataModel, attribute: datamodels.Attribute, value: Any
    ) -> None:
        if bool(value):
            raise ValueError(f"'{attribute.name}' must be equivalent to False")


class ChildModelWithValidators(ModelWithValidators):
    pass


@typing.no_type_check
@pytest.mark.parametrize("model_class", [ModelWithValidators, ChildModelWithValidators])
def test_field_validators(model_class: Type[Union[ModelWithValidators, ChildModelWithValidators]]):

    with pytest.raises(ValueError, match="int_value"):
        model_class(int_value=-1)

    model_class(even_int_value=-2)
    with pytest.raises(ValueError, match="even_int_value"):
        model_class(even_int_value=1)

    model_class(float_value=-2.0)
    with pytest.raises(ValueError, match="float_value"):
        model_class(float_value=4.0)

    with pytest.raises(ValueError, match="str_value"):
        model_class(float_value=1.0, str_value="1.0")

    with pytest.raises(ValueError, match="extra_value"):
        model_class(extra_value=1)

    model_class(extra_value=False)
    model_class(extra_value=0)
    model_class(extra_value="")
    model_class(extra_value=[])
    with pytest.raises(ValueError, match="extra_value"):
        model_class(extra_value=1)


def test_new_field_validators_in_subclass():
    class ChildModelWithValidators(ModelWithValidators):
        @datamodels.validator("int_value")
        def _int_value_validator(self, attribute, value) -> None:
            if value > 10:
                raise ValueError(f"'{attribute.name}' must be lower or equal to 10")

    ChildModelWithValidators(int_value=1)
    with pytest.raises(ValueError, match="int_value"):
        ChildModelWithValidators(int_value=-1)
    with pytest.raises(ValueError, match="int_value"):
        ChildModelWithValidators(int_value=11)


def test_field_validators_in_overwritten_field_in_subclass():
    class ChildModelWithValidators(ModelWithValidators):
        int_value: int = 0

        @datamodels.validator("int_value")
        def _int_value_validator(self, attribute, value) -> None:
            if value > 10:
                raise ValueError(f"'{attribute.name}' must be lower or equal to 10")

    ChildModelWithValidators(int_value=1)
    ChildModelWithValidators(int_value=-1)  # Test new field definition
    with pytest.raises(ValueError, match="int_value"):
        ChildModelWithValidators(int_value=11)

    # Overwrite field definition with a new type
    class AnotherChildModelWithValidators(ModelWithValidators):
        extra_value: float = 0.0

        @datamodels.validator("extra_value")
        def _extra_value_validator(self, attribute, value) -> None:
            if value < 0.0:
                raise ValueError(f"'{attribute.name}' must be a positive number")

    AnotherChildModelWithValidators(extra_value=0.0)
    AnotherChildModelWithValidators(extra_value=1.0)
    with pytest.raises(ValueError, match="extra_value"):
        AnotherChildModelWithValidators(extra_value=-1.0)


class ModelWithRootValidators(datamodels.DataModel):
    int_value: int
    float_value: float
    str_value: str

    class_counter: ClassVar[int] = 0

    @datamodels.root_validator
    @classmethod
    def _root_validator(cls: Type[datamodels.DataModel], instance: datamodels.DataModel) -> None:
        assert cls is type(instance)
        assert issubclass(cls, ModelWithRootValidators)
        assert isinstance(instance, ModelWithRootValidators)
        cls.class_counter = 0

    @datamodels.root_validator
    @classmethod
    def _another_root_validator(
        cls: Type[datamodels.DataModel], instance: datamodels.DataModel
    ) -> None:
        cls = cast(Type[ModelWithRootValidators], cls)
        instance = cast(ModelWithRootValidators, instance)
        assert cls.class_counter == 0
        cls.class_counter += 1

    @datamodels.root_validator
    @classmethod
    def _final_root_validator(
        cls: Type[datamodels.DataModel], instance: datamodels.DataModel
    ) -> None:
        cls = cast(Type[ModelWithRootValidators], cls)
        instance = cast(ModelWithRootValidators, instance)
        assert cls.class_counter == 1
        cls.class_counter += 1

        if instance.int_value == instance.float_value:
            raise ValueError("'int_value' and 'float_value' must be different")


class ChildModelWithRootValidators(ModelWithRootValidators):
    pass


@pytest.mark.parametrize("model_class", [ModelWithRootValidators, ChildModelWithRootValidators])
def test_root_validators(model_class: Type[datamodels.DataModel]):
    model_class(int_value=0, float_value=1.1, str_value="")
    with pytest.raises(ValueError, match="float_value"):
        model_class(int_value=1, float_value=1.0, str_value="")


def test_root_validators_in_subclasses():
    class Model(ModelWithRootValidators):
        @datamodels.root_validator
        @classmethod
        def _root_validator(cls, instance):
            assert cls.class_counter == 2
            cls.class_counter += 10

        @datamodels.root_validator
        @classmethod
        def _another_root_validator(cls, instance):
            assert cls.class_counter == 12
            cls.class_counter += 10

        @datamodels.root_validator
        @classmethod
        def _final_root_validator(cls, instance):
            assert cls.class_counter == 22
            if str(instance.int_value) == instance.str_value:
                raise ValueError("'int_value' and 'str_value' must be different")

    with pytest.raises(ValueError, match="str_value"):
        Model(int_value=1, float_value=0.0, str_value="1")
    with pytest.raises(ValueError, match="float_value"):
        Model(int_value=1, float_value=1.0, str_value="1")
    with pytest.raises(ValueError, match="float_value"):
        Model(int_value=1, float_value=1.0, str_value="1")


# Test field options
def test_field_init():
    @datamodels.datamodel
    class Model:
        hidden_value: int = datamodels.field(init=False)
        value: int
        hidden_value_with_default: int = datamodels.field(default=10, init=False)

    model = Model(value=1)
    assert model.value == 1
    assert model.hidden_value_with_default == 10

    model.hidden_value = -1
    assert model.hidden_value == -1


def test_field_metadata():
    @datamodels.datamodel
    class Model:
        value: int = datamodels.field(metadata={"my_metadata": "META"})

    assert isinstance(Model.__datamodel_fields__.value.metadata, types.MappingProxyType)
    assert Model.__datamodel_fields__.value.metadata["my_metadata"] == "META"


# Test datamodel options
class TestDatamodelOptions:
    def test_frozen(self):
        import attr  # type: ignore[import] # Missing library stubs for Python 3.10)

        @datamodels.datamodel(frozen=True)
        class FrozenModel:
            value: Any = None

        assert FrozenModel.__datamodel_params__.frozen is True
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            FrozenModel().value = 1

        class FrozenModel2(datamodels.DataModel, frozen=True):
            value: Any = None

        assert FrozenModel2.__datamodel_params__.frozen is True
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            FrozenModel2().value = 1

    # Explanation of expected hash() behavior in Python:
    #   https://hynek.me/articles/hashes-and-equality/
    def test_safe_mutable_hash(self):
        string_value = "this is a really long string to avoid string interning 1234567890 +:?.!"

        # Safe mutable case: no hash
        @datamodels.datamodel
        class MutableModel:
            value: Any = None

        with pytest.raises(TypeError, match="unhashable type"):
            hash(MutableModel(value=string_value))

    def test_unsafe_mutable_hash(self):
        # Unsafe mutable case: mutable hash
        @datamodels.datamodel(unsafe_hash=True)
        class MutableModelWithHash:
            value: Any = None

        string_value = "this is a really long string to avoid string interning 1234567890 +:?.!"
        mutable_model = MutableModelWithHash(value=string_value)
        hash_value = hash(mutable_model)
        assert hash_value == hash(MutableModelWithHash(value=string_value))

        # This is the (expected) wrong behaviour with 'unsafe_hash=True',
        # because hash value should not change during lifetime of an object
        mutable_model.value = 42
        assert hash_value != hash(mutable_model)

    def test_safe_frozen_hash(self):
        # Safe frozen case: proper hash
        @datamodels.datamodel(frozen=True)
        class FrozenModel:
            value: Any = None

        string_value = "this is a really long string to avoid string interning 1234567890 +:?.!"

        assert hash(FrozenModel(value=string_value)) == hash(FrozenModel(value=string_value))


# Test module functions
def test_info_functions():
    @datamodels.datamodel
    class Model:
        int_value: int
        float_value: float
        str_value: str
        class_counter: ClassVar[int] = 0

    fields_info = datamodels.fields(Model)
    fields_info_keys = set(fields_info.keys())

    assert isinstance(fields_info, utils.FrozenNamespace)
    assert fields_info_keys == {"int_value", "float_value", "str_value"}
    assert datamodels.get_fields(Model) == fields_info

    model = Model(int_value=1, float_value=2.0, str_value="string")

    assert fields_info == datamodels.fields(model)
    assert datamodels.asdict(model) == {"int_value": 1, "float_value": 2.0, "str_value": "string"}
    assert datamodels.astuple(model) == (1, 2.0, "string")


# Test generic models
CONCRETE_TYPE_SAMPLES: Final = [int, List[float], Tuple[int, ...], Optional[int], Union[int, float]]


@pytest.mark.parametrize("concrete_type", CONCRETE_TYPE_SAMPLES)
def test_generic_model_instantiation_name(concrete_type: Type):
    Model: Type[datamodels.DataModel] = datamodels.concretize(GenericModel, concrete_type)  # type: ignore[misc]
    assert Model.__name__.startswith(GenericModel.__name__)
    assert Model.__name__ != GenericModel.__name__

    Model = datamodels.concretize(GenericModel, concrete_type, class_name="MyNewConcreteClass")  # type: ignore[misc]
    assert Model.__name__ == "MyNewConcreteClass"


@pytest.mark.parametrize("concrete_type", CONCRETE_TYPE_SAMPLES)
def test_generic_model_alias(concrete_type: Type):
    Model: Type[datamodels.DataModel] = datamodels.concretize(GenericModel, concrete_type)  # type: ignore[misc]

    class SubModel(GenericModel[concrete_type]):  # type: ignore  # using run-time type on purpose
        ...

    assert SubModel.__base__ is Model


@pytest.mark.parametrize("concrete_type", CONCRETE_TYPE_SAMPLES)
def test_generic_model_instantiation_cache(concrete_type):
    Model1 = datamodels.concretize(GenericModel, concrete_type)
    Model2 = datamodels.concretize(GenericModel, concrete_type)
    Model3 = datamodels.concretize(GenericModel, concrete_type)

    assert (
        Model1 is Model2
        and Model2 is Model3
        and Model3 is datamodels.concretize(GenericModel, concrete_type)
    )


class TestGenericModelValidation:
    def test_basic_generic_field(self):
        class GenericModel(datamodels.GenericDataModel, Generic[T]):
            value: T

        GenericModel(value=1)
        GenericModel(value="value")
        GenericModel(value=(1.0, "value"))
        GenericModel(value=None)

    def test_partial_generic_field(self):
        class PartialGenericModel(datamodels.GenericDataModel, Generic[T]):
            value: List[T]

        PartialGenericModel(value=[])
        PartialGenericModel(value=[1])
        PartialGenericModel(value=["value"])
        PartialGenericModel(value=[1.0, "value"])
        PartialGenericModel(value=[(1.0, "value")])
        PartialGenericModel(value=[None])
        with pytest.raises(TypeError, match="'PartialGenericModel.value'"):
            PartialGenericModel(value=1)
        with pytest.raises(TypeError, match="'PartialGenericModel.value'"):
            PartialGenericModel(value=(1, 2))

        PartialGenericModel__int = PartialGenericModel[int]
        PartialGenericModel__int(value=[])
        PartialGenericModel__int(value=[1])
        with pytest.raises(TypeError, match="'PartialGenericModel__int.value'"):
            PartialGenericModel__int(value=1)
        with pytest.raises(TypeError, match="'PartialGenericModel__int.value'"):
            PartialGenericModel__int(value=(1, 2))
        with pytest.raises(TypeError, match="'PartialGenericModel__int.value'"):
            PartialGenericModel__int(value=[1.0])
        with pytest.raises(TypeError, match="'PartialGenericModel__int.value'"):
            PartialGenericModel__int(value=["1"])

    def test_partial_specialization(self):
        class PartialGenericModel(datamodels.GenericDataModel, Generic[T, U]):
            value: List[Tuple[T, U]]

        PartialGenericModel(value=[])
        PartialGenericModel(value=[("value", 3)])
        PartialGenericModel(value=[(1, "value")])
        PartialGenericModel(value=[(-1.0, "value")])
        with pytest.raises(TypeError, match="'PartialGenericModel.value'"):
            PartialGenericModel(value=1)
        with pytest.raises(TypeError, match="'PartialGenericModel.value'"):
            PartialGenericModel(value=(1, 2))
        with pytest.raises(TypeError, match="'PartialGenericModel.value'"):
            PartialGenericModel(value=[()])
        with pytest.raises(TypeError, match="'PartialGenericModel.value'"):
            PartialGenericModel(value=[(1,)])

        print(f"{PartialGenericModel.__parameters__=}")
        print(f"{hasattr(PartialGenericModel ,'__args__')=}")

        PartiallySpecializedGenericModel = PartialGenericModel[int, U]
        print(f"{PartiallySpecializedGenericModel.__datamodel_fields__=}")
        print(f"{PartiallySpecializedGenericModel.__parameters__=}")
        print(f"{PartiallySpecializedGenericModel.__args__=}")

        PartiallySpecializedGenericModel(value=[])
        PartiallySpecializedGenericModel(value=[(1, 2)])
        PartiallySpecializedGenericModel(value=[(1, "value")])
        PartiallySpecializedGenericModel(value=[(1, (11, 12))])
        with pytest.raises(TypeError, match=".value'"):
            PartiallySpecializedGenericModel(value=1)
        with pytest.raises(TypeError, match=".value'"):
            PartiallySpecializedGenericModel(value=(1, 2))
        with pytest.raises(TypeError, match=".value'"):
            PartiallySpecializedGenericModel(value=[1.0])
        with pytest.raises(TypeError, match=".value'"):
            PartiallySpecializedGenericModel(value=["1"])

        # TODO(egparedes): after fixing partial nested datamodel specialization
        # noqa: e800 FullySpecializedGenericModel = PartiallySpecializedGenericModel[str]
        # noqa: e800 print(f"{FullySpecializedGenericModel.__datamodel_fields__=}")
        # noqa: e800 print(f"{FullySpecializedGenericModel.__parameters__=}")
        # noqa: e800 print(f"{FullySpecializedGenericModel.__args__=}")

        # noqa: e800 FullySpecializedGenericModel(value=[])
        # noqa: e800 FullySpecializedGenericModel(value=[(1, "value")])
        # noqa: e800 with pytest.raises(TypeError, match=".value'"):
        # noqa: e800     FullySpecializedGenericModel(value=1)
        # noqa: e800 with pytest.raises(TypeError, match=".value'"):
        # noqa: e800     FullySpecializedGenericModel(value=(1, 2))
        # noqa: e800 with pytest.raises(TypeError, match=".value'"):
        # noqa: e800     FullySpecializedGenericModel(value=[1.0])
        # noqa: e800 with pytest.raises(TypeError, match=".value'"):
        # noqa: e800     FullySpecializedGenericModel(value=["1"])
        # noqa: e800 with pytest.raises(TypeError, match=".value'"):
        # noqa: e800     FullySpecializedGenericModel(value=1)
        # noqa: e800 with pytest.raises(TypeError, match=".value'"):
        # noqa: e800     FullySpecializedGenericModel(value=[(1, 2)])
        # noqa: e800 with pytest.raises(TypeError, match=".value'"):
        # noqa: e800     FullySpecializedGenericModel(value=[(1, (11, 12))])

    # Reuse sample_type_data from test_field_type_hint
    @pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], SAMPLE_TYPE_DATA)
    def test_concrete_field_type(
        self, type_hint: str, valid_values: Sequence[Any], wrong_values: Sequence[Any]
    ):
        concrete_type: Type = eval(type_hint)
        Model: Type[datamodels.DataModel] = GenericModel[concrete_type]  # type: ignore[valid-type,misc,assignment]  # concrete_type

        for value in valid_values:
            Model(value=value)

        for value in wrong_values:
            with pytest.raises((TypeError, ValueError), match=".value'"):
                Model(value=value)


# ---- Validators ----
def test_non_empty():
    @datamodels.datamodel
    class Model:
        str_value: str = datamodels.field(validator=datamodels.validators.non_empty())
        list_value: List[int] = datamodels.field(validator=datamodels.validators.non_empty())

    Model("AAAA", [1])

    with pytest.raises(ValueError, match="Empty"):
        Model("", [])

    with pytest.raises(ValueError, match="Empty"):
        Model("", [1])

    with pytest.raises(ValueError, match="Empty"):
        Model("AAAA", [])
