# -*- coding: utf-8 -*-

from numpy import float32, float64, int32, int64

from functional.common import Field


__all__ = ["Field", "float32", "float64", "int32", "int64"]

TYPE_BUILTINS = [Field, float, float32, float64, int, int32, int64, bool, tuple]
TYPE_BUILTIN_NAMES = [t.__name__ for t in TYPE_BUILTINS]
