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
The command will take a python file and the name of one of the available backends as input and output
backend specific source code. Depending on the backend and other options this might be python, c++,
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

The usage is explained best using a small example.

Assume the following file structure:

.. code-block:: bash

   $ tree .
   pwd
   ├── constants.py
   └── stencils.py

`stencils.py` contains the `gtscript` code to be compiled to stencils. The contents might look something like the following example.

.. code-block:: python

   ## use-dsl: gtscript

   from .constants import PI


   @function
   def square(inp_field):
      return inp_field * inp_field


   @stencil
   def stencil_a(inp_field: Field[float64], out_field: Field[float64]):
      with computation(PARALLEL), interval(...):
         out_field = square(inp_field)


   @stencil
   def stencil_b(inp_field: Field[float64], out_field: Field[float64]):
      from __externals__ import COMPILE_TIME_VALUE
      with computation(PARALLEL), interval(...):
         out_field = PI * inp_field + COMPILE_TIME_VALUE

Notice that this file uses names from `gt4py.gtscript` without importing `gt4py`. The names will be injected by
`gtpyc` upon recognizing the `## use-dsl: gtscript` comment.
Also note that `stencil_b` uses an external value which is not available in the file itself, so it 
will have to be supplied on the command line.
The file `constants.py` contains some constant values (which might be templated by the build system).

In order to get C++ code we can now run `gtpyc` with for example the `GridTools` multi core backend (`-b gtmc`) and
tell it to generate the stencils in the new subdirectory `stencils` (`-o stencils`). 

.. code-block:: bash

   $ gtpyc -b gtmc stencils.py -o stencils -e COMPILE_TIME_VALUE 
   $ tree .stencils/
   stencils
   ├── stencil_a.cpp
   ├── stencil_a.hpp
   ├── stencil_b.cpp
   └── stencil_b.hpp

The current backends of `gt4py` (with the exception of the python-only ones) all have the ability to generate python bindings.
Future backends might allow bindings for other languages. This is accessible through an additional CLI option, which should
be validated based on the chosen backend.

.. code-block:: bash

   $ gtpyc -b gtx86 stencils.py -o stencils --bindings=python -e COMPILE_TIME_VALUE 
   $ tree .stencils/
   stencils
   ├── stencil_a_bindings.cpp
   ├── stencil_a.cpp
   ├── stencil_a.hpp
   ├── stencil_a.py
   ├── stencil_b_bindings.cpp
   ├── stencil_b.cpp
   ├── stencil_b.hpp
   └── stencil_b.py

Finally, the backend may allow options specific to it. These can be passed using the `--option` or `-O` flag.
For example the `GridTools` multi core backend takes a `debug` flag (which does nothing during source file generation) but
would activate debug flags if we ask gt4py to compile a readily importable python extension.

.. code-block:: bash

   $ gtpyc -b gtmc stencils.py -o stencils -e COMPILE_TIME_VALUE -O debug True --bindings=python --compile-bindings
   $ tree .stencils/
   stencils
   ├── stencil_a_bindings.cpp
   ├── stencil_a.cpp
   ├── stencil_a.hpp
   ├── _stencil_a.so  # compiled with debug flags
   ├── stencil_a.py
   ├── stencil_b_bindings.cpp
   ├── stencil_b.cpp
   ├── stencil_b.hpp
   ├── _stencil_b.so  # compiled with debug flags
   └── stencil_b.py

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

Any description of design ideas and implementation refers to the
`reference implementation <https://github.com/GridTools/gt4py/pull/23>`_.
This section will be updated as the reference implementation progresses.

Enabling all of gtscript without importing from gt4py
+++++++++++++++++++++++++++++++++++++++++++++++++++++

The currently chosen route for this is to require a comment at the very start of the file::

   ## use-dsl: gtscript

This will serve two purposes, first it will mark the file as being written in gtscript.
Any name that in python can be accessed by `from gt4py.gtscript import *` will work when
compiling with `gtpyc` but will be deemed undefined by the python interpreter.
It is not planned to provide any means of informing python syntax checkers to consider
these names as defined.
Secondly `gtpyc` can replace this line with an actual `import` line without changing line numbers
for error messages.

Obviously, some symbols like the `@stencil` decorator will have to be either changed or
monkey patched, since we do not want loading of the input gtscript file to already trigger
a compilation and though we might want to give default arguments to the backend in the decorator
we want to be able to override them on the CLI.

Importing python modules
++++++++++++++++++++++++

Gtscript files can import python modules as well as gtscript files only via relative imports in the
format of `from .<module> import <name>`. `gtpyc` will make sure of turning all gtscript imports into valid
python code before importing them.

Passing externals
+++++++++++++++++

There are two supported ways to configure values at compile / generate time.

 * By relative import of a python file, which may be automatically generated from a template.
   The latter could happen as part of a build system depending on build parameters. In this case
   the stencil definition can use the values without importing them from `__externals__`. If it does, however,
   the external value can be overriden on the command line using the following second option.
 * By passing externals options on the command line. In this case the external will be passed
   to every stencil in this run of `gtpyc` and each stencil needs to import it from `__externals__` to use it.

Generating Language bindings
++++++++++++++++++++++++++++

The intention of this GDP is to support generating language bindings for all languages the chosen backend
supports. These language bindings are intended to be usable without `gt4py` as a requirement. This is important
to allow usage of generated bindings in non-GPL3 projects.

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

The discussion around the reference implementation is located in it's separate
`pull request <https://github.com/GridTools/gt4py/pull/23>`_.


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
.. _reference_impl_pr: https://github.com/GridTools/gt4py/pull/23


Copyright
---------

This document has been placed in the public domain. [1]_
