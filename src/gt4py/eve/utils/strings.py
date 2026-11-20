# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""String utilities."""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import itertools
import re
from collections.abc import Callable, Iterable, Iterator
from typing import Union


AnyWordsIterable = Union[str, Iterable[str]]


class CaseStyleConverter:
    """Utility class to convert name strings to different case styles.

    Functionality exposed through :meth:`split()`, :meth:`join()` and
    :meth:`convert()` methods.

    """

    class CASE_STYLE(enum.Enum):
        CONCATENATED = "concatenated"
        CANONICAL = "canonical"
        CAMEL = "camel"
        PASCAL = "pascal"
        SNAKE = "snake"
        KEBAB = "kebab"

    @classmethod
    def split(cls, name: str, case_style: Union[CASE_STYLE, str]) -> list[str]:
        if isinstance(case_style, str):
            case_style = cls.CASE_STYLE(case_style)
        assert isinstance(case_style, cls.CASE_STYLE)
        if case_style == cls.CASE_STYLE.CONCATENATED:
            raise ValueError("Impossible to split a simply concatenated string")

        splitter: Callable[[str], list[str]] = getattr(cls, f"split_{case_style.value}_case")
        return splitter(name)

    @classmethod
    def join(cls, words: AnyWordsIterable, case_style: Union[CASE_STYLE, str]) -> str:
        if isinstance(case_style, str):
            case_style = cls.CASE_STYLE(case_style)
        assert isinstance(case_style, cls.CASE_STYLE)
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, collections.abc.Iterable):
            raise TypeError(f"'{words}' type is not a valid sequence of words")

        joiner: Callable[[AnyWordsIterable], str] = getattr(cls, f"join_{case_style.value}_case")
        return joiner(words)

    @classmethod
    def convert(
        cls, name: str, source_style: Union[CASE_STYLE, str], target_style: Union[CASE_STYLE, str]
    ) -> str:
        return cls.join(cls.split(name, source_style), target_style)

    # Following `join_...`` functions are based on:
    #    https://blog.kangz.net/posts/2016/08/31/code-generation-the-easier-way/
    #
    @staticmethod
    def join_concatenated_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "".join(words).lower()

    @staticmethod
    def join_canonical_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return (" ".join(words)).lower()

    @staticmethod
    def join_camel_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else list(words)
        return words[0].lower() + "".join(word.title() for word in words[1:])

    @staticmethod
    def join_pascal_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "".join(word.title() for word in words)

    @staticmethod
    def join_snake_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "_".join(words).lower()

    @staticmethod
    def join_kebab_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "-".join(words).lower()

    # Following `split_...`` functions are based on:
    #    https://stackoverflow.com/a/29920015/7232525
    #
    @staticmethod
    def split_canonical_case(name: str) -> list[str]:
        return name.split()

    @staticmethod
    def split_camel_case(name: str) -> list[str]:
        matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", name)
        return [m.group(0) for m in matches]

    split_pascal_case = split_camel_case

    @staticmethod
    def split_snake_case(name: str) -> list[str]:
        return name.split("_")

    @staticmethod
    def split_kebab_case(name: str) -> list[str]:
        return name.split("-")


@dataclasses.dataclass(frozen=True)
class SequentialIDGenerator:
    """Simple sequential ID generator."""

    prefix: str = ""
    counter: Iterator[int] = dataclasses.field(default_factory=itertools.count)
    #: A string to be used as template for the new ids.
    #: It should contain the `{prefix}`and `{id}` format keys.
    format: str = "{prefix}_{id}"

    def __next__(self) -> str:
        return self.format.format(prefix=self.prefix, id=next(self.counter))

    def next(self) -> str:
        return self.__next__()
