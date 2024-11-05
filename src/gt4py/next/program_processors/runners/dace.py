# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import factory

from gt4py.next import backend
from gt4py.next.program_processors.runners.dace_fieldview import workflow as dace_fieldview_workflow
from gt4py.next.program_processors.runners.dace_iterator import workflow as dace_iterator_workflow
from gt4py.next.program_processors.runners.gtfn import GTFNBackendFactory


class DaCeIteratorBackendFactory(GTFNBackendFactory):
    class Params:
        otf_workflow = factory.SubFactory(
            dace_iterator_workflow.DaCeWorkflowFactory,
            device_type=factory.SelfAttribute("..device_type"),
            use_field_canonical_representation=factory.SelfAttribute(
                "..use_field_canonical_representation"
            ),
        )
        auto_optimize = factory.Trait(
            otf_workflow__translation__auto_optimize=True, name_postfix="_opt"
        )
        use_field_canonical_representation: bool = False

    name = factory.LazyAttribute(
        lambda o: f"run_dace_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}.itir"
    )

    transforms = backend.LEGACY_TRANSFORMS


run_dace_cpu = DaCeIteratorBackendFactory(cached=True, auto_optimize=True)
run_dace_cpu_noopt = DaCeIteratorBackendFactory(cached=True, auto_optimize=False)

run_dace_gpu = DaCeIteratorBackendFactory(gpu=True, cached=True, auto_optimize=True)
run_dace_gpu_noopt = DaCeIteratorBackendFactory(gpu=True, cached=True, auto_optimize=False)

itir_cpu = run_dace_cpu
itir_gpu = run_dace_gpu


class DaCeFieldviewBackendFactory(GTFNBackendFactory):
    class Params:
        otf_workflow = factory.SubFactory(
            dace_fieldview_workflow.DaCeWorkflowFactory,
            device_type=factory.SelfAttribute("..device_type"),
            auto_optimize=factory.SelfAttribute("..auto_optimize"),
        )
        auto_optimize = factory.Trait(name_postfix="_opt")

    name = factory.LazyAttribute(
        lambda o: f"run_dace_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}.gtir"
    )

    transforms = backend.DEFAULT_TRANSFORMS


gtir_cpu = DaCeFieldviewBackendFactory(cached=True, auto_optimize=False)
gtir_gpu = DaCeFieldviewBackendFactory(gpu=True, cached=True, auto_optimize=False)
