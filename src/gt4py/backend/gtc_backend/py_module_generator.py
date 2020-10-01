from gt4py.backend.base import BaseModuleGenerator


class GTCPyModuleGenerator(BaseModuleGenerator):
    """Generate a python stencil module loadable by gt4py."""

    def generate_implementation(self) -> str:
        return "pass"
