# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.otf import compiled_program

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import simple_mesh


def test_offset_provider_to_type_unsafe():
    mesh = simple_mesh(None)
    offset_provider = mesh.offset_provider

    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_clear()

    compiled_program._offset_provider_to_type_unsafe(offset_provider)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 1
    compiled_program._offset_provider_to_type_unsafe(offset_provider)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 1
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().hits == 1

    offset_provider2 = {"V2E": mesh.offset_provider["V2E"]}
    compiled_program._offset_provider_to_type_unsafe(offset_provider2)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 2
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().hits == 1
