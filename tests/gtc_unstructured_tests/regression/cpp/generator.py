import sys
import importlib
import os

from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator
from gtc_unstructured.frontend.frontend import GTScriptCompilationTask
from gtc_unstructured.irs.icon_bindings_codegen import IconBindingsCodegen


def _generate(stencil, *, output_dir, stencil_name="stencil", generate_icon=False, mode="unaive"):
    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
        extension = ".cc"
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator
        extension = ".cu"

    compilation_task = GTScriptCompilationTask(stencil)
    generated_code = compilation_task.generate(debug=False, code_generator=code_generator)

    output_file = f"{output_dir}/generated_{stencil_name}_{mode}.hpp"
    with open(output_file, "w+") as output:
        output.write(generated_code)

    if generate_icon:
        icon_bindings = IconBindingsCodegen().apply(
            compilation_task.gtir, stencil_code=generated_code
        )
        output_file = f"{output_dir}/generated_icon_{stencil_name}_{mode}{extension}"
        with open(output_file, "w+") as output:
            output.write(icon_bindings)


def main():
    if len(sys.argv) <= 2:
        print(
            f'Usage: {sys.argv[0]} <stencil_definition_module> <output_dir> ["ugpu"/"unaive" "icon"]'
        )
        exit(1)
    module_name = sys.argv[1]
    output_dir = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "unaive"
    generate_icon = len(sys.argv) > 4 and sys.argv[4] == "icon"

    module = importlib.import_module(module_name)
    _generate(
        module.sten,
        stencil_name=module_name,
        mode=mode,
        output_dir=output_dir,
        generate_icon=generate_icon,
    )


if __name__ == "__main__":
    main()


def default_main(sten):
    output_dir = os.path.dirname(os.path.realpath(__file__))
    _generate(sten, output_dir=output_dir, generate_icon=True, mode="unaive")
    _generate(sten, output_dir=output_dir, generate_icon=True, mode="ugpu")
