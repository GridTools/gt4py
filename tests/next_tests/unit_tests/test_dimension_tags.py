# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import re

import pytest

from gt4py.next import dimension_tags


class TestCodegenName:
    """`codegen_name` must be injective: a collision means two dimensions sharing a symbol."""

    @pytest.mark.parametrize(
        "tag, expected",
        [
            ("IDim", "IDim"),
            ("mod.V2E.Local", "mod_dV2E_dLocal"),
            ("mod:V2E.Local", "mod_cV2E_dLocal"),
            ("a.b_c", "a_db_uc"),
            ("a_b.c", "a_ub_dc"),
            ("my__mod.X", "my_u_umod_dX"),
            ("_CONST_DIM", "_uCONST_uDIM"),
            # a parametrized tag: the brackets must not survive into the identifier
            ("gt4py.next.common:Staggered[pkg:K]", "gt4py_dnext_dcommon_cStaggered_lpkg_cK_r"),
        ],
    )
    def test_known_values(self, tag, expected):
        assert dimension_tags.codegen_name(tag) == expected
        assert dimension_tags.from_codegen_name(expected) == tag

    @pytest.mark.parametrize(
        "tag",
        [
            "_u",
            "_d",
            "_c",
            "_l",
            "_r",
            "a_ud.b",
            "_ud_du",
            "..",
            "__",
            "[]",
            "a[b.c]",
            "_l[_r]",
            "a:_c",
        ],
    )
    def test_roundtrip_adversarial(self, tag):
        """Tags that look like the escape sequences themselves must still round-trip."""
        assert dimension_tags.from_codegen_name(dimension_tags.codegen_name(tag)) == tag

    def test_output_is_a_valid_identifier(self):
        for tag in ["mod.V2E.Local", "a.b_c", "_CONST_DIM", "pkg.sub.Dim", "mod.Staggered[mod.K]"]:
            assert re.fullmatch(r"[A-Za-z_]\w*", dimension_tags.codegen_name(tag)), tag

    def test_injective_and_reversible_exhaustively(self):
        """
        Exhaustive over the characters that can actually collide, to a length that covers
        every interaction between them.

        The naive scheme -- `_` -> `__` then `.` -> `_` -- fails this with 686 collisions,
        because a dot becomes a single underscore and `'..'` collides with an escaped `'_'`.
        """
        # every character a tag can contain that needs escaping, plus the escape letters
        alphabet = "a._:[]udclr"
        seen: dict[str, str] = {}
        for length in range(1, 5):
            for tag in map("".join, itertools.product(alphabet, repeat=length)):
                mangled = dimension_tags.codegen_name(tag)
                assert mangled not in seen, (
                    f"collision: {seen.get(mangled)!r} and {tag!r} both map to {mangled!r}"
                )
                seen[mangled] = tag
                assert dimension_tags.from_codegen_name(mangled) == tag
        assert len(seen) == sum(len(alphabet) ** n for n in range(1, 5))
