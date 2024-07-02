# Copyright (C) 2022, Benjamin Drung <bdrung@posteo.de>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import contextlib
import typing
import unittest.mock

@contextlib.contextmanager
def wrap_object(
    target: object, attribute: str
) -> typing.Generator[unittest.mock.MagicMock, None, None]:
    """Wrap the named member on an object with a mock object.

    wrap_object() can be used as a context manager. Inside the
    body of the with statement, the attribute of the target is
    wrapped with a :class:`unittest.mock.MagicMock` object. When
    the with statement exits the patch is undone.

    The instance argument 'self' of the wrapped attribute is
    intentionally not logged in the MagicMock call. Therefore
    wrap_object() can be used to check all calls to the object,
    but not differentiate between different instances.
    """
    mock = unittest.mock.MagicMock()
    real_attribute = getattr(target, attribute)

    def mocked_attribute(self, *args, **kwargs):
        mock.__call__(*args, **kwargs)
        return real_attribute(self, *args, **kwargs)

    with unittest.mock.patch.object(target, attribute, mocked_attribute):
        yield mock
