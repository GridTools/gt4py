# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import factory
import pytest

import factory

from gt4py.next import factory_utils


@dataclasses.dataclass
class Person:
    name: str
    nickname: str


class PersonFactory(factory.Factory):
    class Meta:
        model = Person

    class Params:
        endearment: str = True

    name = "Joe"

    @factory_utils.dynamic_transformer(default=factory.SelfAttribute(".name"))
    def nickname(self, nickname):
        if self.endearment:
            name = f"{nickname}y"
        return name


def test_transformer_applies_transform_to_default_and_override():
    # default value is transformed
    person = PersonFactory()
    assert person.nickname == "Joey"

    # overridden `name` value is also transformed
    john = PersonFactory(name="John")
    assert john.nickname == "Johny"
