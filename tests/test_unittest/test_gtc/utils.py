import re
from typing import Literal, Pattern, Union


def match(value: str, regexp: "Union[str, Pattern]") -> "Literal[True]":
    """
    Stolen from `pytest.raises`.
    Check whether the regular expression `regexp` matches `value` using :func:`python:re.search`.
    If it matches `True` is returned.
    If it doesn't match an `AssertionError` is raised.
    """
    assert re.search(regexp, str(value)), "Pattern {!r} does not match {!r}".format(
        regexp, str(value)
    )
    return True
