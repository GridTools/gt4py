# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any

from gt4py.next.otf import languages
from gt4py.next.program_processors import otf_compile_executor
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn_cpu


CPP_WITH_CUDA = languages.LanguageWithHeaderFilesSettings(
    formatter_key="cpp",
    formatter_style="llvm",
    file_extension="cpp.cu",
    header_extension="hpp",
)


gtfn_gpu: otf_compile_executor.OTFCompileExecutor[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python, Any
] = otf_compile_executor.OTFCompileExecutor(
    name="gpu_backend",
    otf_workflow=gtfn_cpu.run_gtfn.otf_workflow.replace(
        translation=gtfn_module.GTFNTranslationStep(
            language_settings=CPP_WITH_CUDA, gtfn_backend=gtfn_module.GTFNBackendKind.GPU
        ),
    ),
)
