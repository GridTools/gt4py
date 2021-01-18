# -*- coding: utf-8 -*-
import re
from typing import Pattern, Union


def match(value: str, regexp: "Union[str, Pattern]") -> bool:
    """
    Check that `regex` pattern matches `value`.

    Stolen from `pytest.raises`.
    Check whether the regular expression `regexp` matches `value` using :func:`python:re.search`.
    If it matches `True` is returned.
    If it doesn't match an `AssertionError` is raised.
    """
    assert re.search(regexp, str(value)), "Pattern {!r} does not match {!r}".format(
        regexp, str(value)
    )
    return True
