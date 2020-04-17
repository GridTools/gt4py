=================================
GDP 1 — Standalone Compiler CLI
=================================

:Author: Rico Häuselmann <ricoh@cscs.ch>
:Status: Draft
:Type: Feature
:Created: 17.04.2020
:Discussion PR: <PR url>


Abstract
--------

GT4Py currently provides a python API / embedded DSL for defining,
compiling and running GridTools stencils from python programs.

This GDP describes an additional CLI which can be integrated into the build
process of non-python programs to compile, link and run stencils written
in GT4Py DSL from any host language that can link to C++ libraries.

Motivation and Scope
--------------------

This GDP proposes adding a CLI command named `gtpyc` (naming rationale documented somewhere below).
The command will take a python file and the name of the GridTools backend as input and output a
backend specific code file. Depending on the backend and other options this might be python, c++,
cuda or object code (This part of the design is very much WIP as of now).

In order to support the usecases listed below the input python files must be written in
gtscript DSL without explicitly importing GT4Py. Instead of the decorators provided by the
`gtscript` module, comments may be used, or functions may be inferred to be a `stencil` or `submodule`
depending on whether they return something or not. In order to stay maintainable `gtpyc`
will at most inject names from the `gtscript` module into the namespace of the input python module
before running the input code through the same process applied to on-line compilation of stencils
and submodules.

Other language projects
+++++++++++++++++++++++

In order to benefit from the higher abstraction level of the GT4Py eDSL it should
not be required to run python code at runtime. Especially for existing programs
written in other languages it makes more sense to link to libraries created by GT4Py
as part of the build process of the host program.

Avoiding runtime dependency
+++++++++++++++++++++++++++

Even for python projects it may be desirable to distribute only the extension
modules created by GT4Py, not the code that generated them,
thus requiring the end user to only install GridTools, not GT4Py.

Licensing
+++++++++

Using only the GT4Py generated stencils in a project without depending on GT4Py at runtime
allows to use a licence other than GPL3 in said project without the express permission of CSCS.

Usage and Impact
----------------

This section describes how users of GT4Py will use features described in this
GDP. It should be comprised mainly of code examples that wouldn't be possible
without acceptance and implementation of this GDP, as well as the impact the
proposed changes would have on the ecosystem. This section should be written
from the perspective of the users of GT4Py, and the benefits it will provide
them; and as such, it should include implementation details only if
necessary to explain the functionality.

Backward compatibility
----------------------

This section describes the ways in which the GDP breaks backward compatibility.


Detailed description
--------------------

This section should provide a detailed description of the proposed change.
It should include examples of how the new functionality would be used,
intended use-cases and pseudo-code illustrating its use.


Related Work
------------

This section should list relevant and/or similar technologies, possibly in other
libraries. It does not need to be comprehensive, just list the major examples of
prior and relevant art.


Implementation
--------------

This section lists the major steps required to implement the GDP.  Where
possible, it should be noted where one step is dependent on another, and which
steps may be optionally omitted.  Where it makes sense, each step should
include a link to related pull requests as the implementation progresses.

Any pull requests or development branches containing work on this GDP should
be linked to from here.  (A GDP does not need to be implemented in a single
pull request if it makes sense to implement it in discrete phases).


Alternatives
------------

If there were any alternative solutions to solving the same problem, they should
be discussed here, along with a justification for the chosen approach.


Discussion
----------

This section may just be a bullet list including links to any discussions
regarding the GDP:

- This includes links to relevant GitHub issues and publicly available discussions.


References and Footnotes
------------------------

.. [1] Each GDP must either be explicitly labeled as placed in the public domain (see
   this GDP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
