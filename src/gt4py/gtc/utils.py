import collections.abc

from devtools import debug


def flatten_list(nested_iterables, filter_none=False, *, skip_types=(str, bytes)):
    return list(flatten_list_iter(nested_iterables, filter_none, skip_types=skip_types))


def flatten_list_iter(nested_iterables, filter_none=False, *, skip_types=(str, bytes)):
    for item in nested_iterables:
        if isinstance(item, list) and not isinstance(item, skip_types):
            yield from flatten_list(item, filter_none)
        else:
            if item is not None or not filter_none:
                yield item


class ListTuple(collections.abc.Sequence):
    def __init__(self, *args):
        assert all([isinstance(arg, list) for arg in args])
        self.tpl = args

    def __len__(self):
        return len(self.tpl)

    def __getitem__(self, key):
        return self.tpl[key]

    def __add__(self, other):
        if isinstance(other, ListTuple):
            assert len(self.tpl) == len(other.tpl)
            return ListTuple(self.tpl[0] + other.tpl[0], self.tpl[1] + other.tpl[1])
        elif isinstance(other, list):
            flattened = flatten_list(other)
            if all([isinstance(elem, ListTuple) for elem in flattened]):
                for elem in flattened:
                    self += elem
                return self
            else:
                raise ValueError(
                    "Can only concat ListTuples and lists of ListTuples, got list of something else.".format(
                        type(other).__name__
                    )
                )
        else:
            raise ValueError(
                "Can only concat ListTuples and lists of ListTuples, got {}.".format(
                    type(other).__name__
                )
            )
