# -*- coding: utf-8 -*-
"""Stencil Object that allows for deferred building."""
from typing import TYPE_CHECKING, Any, Dict

from cached_property import cached_property


if TYPE_CHECKING:
    from gt4py.backend.base import Backend
    from gt4py.stencil_builder import StencilBuilder
    from gt4py.stencil_object import StencilObject


class LazyStencil:
    """
    A stencil object which defers compilation until it is needed.

    Usually obtained using the :py:func:`gt4py.gtscript.lazy_stencil` decorator, not directly
    instanciated.
    This is done by keeping a reference to a :py:class:`gt4py.stencil_builder.StencilBuilder`
    instance.

    Compilation happens implicitly on first access to the `implementation` property.
    Low-level build utilities are accessible through the public :code:`builder` attribute.
    """

    def __init__(self, builder: "StencilBuilder"):
        self.builder = builder

    @cached_property
    def implementation(self) -> "StencilObject":
        """
        Expose the compiled backend-specific python callable which executes the stencil.

        Compilation happens at first access, the result is cached and should consecutively be
        accessible without overhead (not rigorously tested / benchmarked).
        """
        impl = self.builder.build()()
        return impl

    @property
    def backend(self) -> "Backend":
        """Do not trigger a build."""
        return self.builder.backend

    @property
    def field_info(self) -> Dict[str, Any]:
        """
        Access the compiled stencil object's `field_info` attribute.

        Triggers a build if necessary.
        """
        return self.implementation.field_info

    def check_syntax(self) -> None:
        """Create the gtscript IR for the stencil, failing on syntax errors."""
        if not self.builder.definition_ir:
            raise RuntimeError("Frontend did not raise a syntax error but did not generate IR.")

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Execute the stencil, building the stencil if necessary."""
        self.implementation(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Pass through to the implementation.run."""
        self.implementation.run(*args, **kwargs)
