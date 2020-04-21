=================================
GDP 1 — Standalone Compiler CLI
=================================

:Author: Rico Häuselmann <ricoh@cscs.ch>
:Status: Draft
:Type: Feature
:Created: 17.04.2020
:Discussion PR: `<https://github.com/GridTools/gt4py/pull/21>`


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

Limited Scope
+++++++++++++

In order to support the usecases listed below the input python files must be written in
gtscript DSL without explicitly importing GT4Py. Instead of the decorators provided by the
`gtscript` module, comments may be used, or functions may be inferred to be a `stencil` or `submodule`
depending on whether they return something or not. In order to stay maintainable `gtpyc`
will not add any new logic beyond reading input python code and command line options.

Possible exceptions might be 
 * The minimum amount of logic to distinguish subroutines from stencils in the input python code
 * A mechanism to decide via CLI option whether to create python extensions or not

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

The usage is envisioned as follows::

   $ gtpyc -b mc stencils.py -o stencils
   $ ls
   > stencils.py stencils.so

builds a shared library from python source
::

   $ gtpyc -b mc -f cpp stencils.py -o stencils
   $ ls
   > stencils.py stencils.cpp stencils.hpp

builds GT code from python source
::

   $ gtpyc -b mc -f py stencils.py -o stencils_module
   $ ls
   > stencils.py stencils_module.so stencils_module.py

builds a python extension

Additional Commandline options will mostly correspond to the keyword arguments of
the `gtscript.stencil` decorator.

This should be easy to incorporate into existing build systems as an additional
step from `.py` source files to `.cpp` or `.cu` sources before building and linking
or as an alternative step to build `.py` sources into ready to link libraries.

Backward compatibility
----------------------

This GDP is aimed to be fully backward-compatible.


Detailed description
--------------------

This section will be updated as the reference implementation progresses.

Related Work
------------

CLIs of well-known compilers (Provide CLI conventions):
 * `clang`_
 * `gcc`_
 * `gfortran`_

Implementation
--------------

Implementation will start with a proof-of-concept CLI with an absolutely mninimal
feature set, taking a single function in an input `.py` file and outputting
the result of the stencil compilation in a separate file.

If it becomes apparent at that stage that changes to the internal structure
would become necessary these will likely be treated in separate GDPs.

The PoC will utilize the `click`_ framework for the CLI, since it encourages
separation and reuse of CLI argument / option handling and documentation code
from program logic. None of the known limitations of `click`_ are foreseen to
be detrimental to what this GDP wants to achieve.

Reasons for choosing `click`_
+++++++++++++++++++++++++++++
 * separation of concerns
 * ease of reuse of CLI components
 * built in command completion for bash, zsh etc
 * built-in testing api


Alternatives
------------

Using `argparse` for the CLI
++++++++++++++++++++++++++++

Using `argparse`_ has been rejected. although it is not impossible to separate
option handling code from program logic, any attempt to do so consistently would
lead to partially reinventing one of the more advanced frameworks like `click`_.

The author of this GDP does believe the additional requirement of a small
pure-python framework like `click`_ to be outweighed by the benefits.

Discussion
----------

The discussion for this GDP will be in the draft PR for it, which is to be found
`here <https://github.com/GridTools/gt4py/pull/21>`_.


References and Footnotes
------------------------

.. [1] Each GDP must either be explicitly labeled as placed in the public domain (see
   this GDP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. _click: https://click.palletsprojects.com/en/7.x/
.. _argparse: https://docs.python.org/3/library/argparse.html
.. _clang: https://clang.llvm.org/docs/ClangCommandLineReference.html
.. _gcc: https://gcc.gnu.org/onlinedocs/gcc/Invoking-GCC.html
.. _gfortran: https://gcc.gnu.org/onlinedocs/gfortran/Invoking-GNU-Fortran.html#Invoking-GNU-Fortran
.. _discussion_pr: https://github.com/GridTools/gt4py/pull/21


Copyright
---------

This document has been placed in the public domain. [1]_
