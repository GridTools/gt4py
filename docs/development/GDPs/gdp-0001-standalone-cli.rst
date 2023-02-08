=================================
GDP 1 — Standalone Compiler CLI
=================================

:Author: Rico Häuselmann <ricoh@cscs.ch>
:Status: Accepted
:Type: Feature
:Created: 17.04.2020
:Discussion PR: `https://github.com/GridTools/gt4py/pull/21 <discussion_pr>`_
:Implementation: `https://github.com/GridTools/gt4py/pull/23 <reference_impl_pr>`_


Abstract
--------

GT4Py currently provides a Python API / embedded DSL for defining,
compiling and running GridTools stencils from Python programs.

This GDP describes an additional CLI which can be integrated into the build
process of non-Python programs to compile, link and run stencils written
in GT4Py DSL (GTScript) from any host language that can link to C++ libraries.

A `reference implementation <reference_impl_pr>`_ exists, which will on acception of this GDP be
drawn on for adding functionality of this GDP in a series of separate pull requests.

Motivation and Scope
--------------------

This GDP proposes adding a CLI command named ``gtpyc`` (naming rationale / alternatives documented
somewhere below). The command will take a GTScript file and the name of one of the available
backends as input and output backend specific source code. This includes any language bindings
supported by the backend. Commandline options will allow full control over what bindings should be
created if any, depending on what the backend supports.

A GTScript file denotes a file with a to-be-defined extension (suggestion: ``.gt.py``), beginning
with the line 

.. code-block:: python

   # [GT] using-dsl: gtscript 
   
All other content must be valid Python under the assumption that the first line has been replaced
by :code:`from gt4py.cartesian.gtscript import *`.

In support of this, two more features are proposed:

* A mechanism to allow GTScript files as if they were Python modules.  The usage will be
  :code:`from gt4py.cartesian import gtpy_import; gtpy_import.install()`.

* A lazy variant or replacement of the ``stencil`` decorator, returning an object that supports
  manual stepwise compilation.

Limited Scope
+++++++++++++

In order to support the usecases listed below the input Python files must be written in GTScript
DSL without explicitly importing GT4Py. Instead of the decorators provided by the ``gtscript``
module, comments may be used, or functions may be inferred to be a ``stencil`` or ``submodule``
depending on whether they return something or not. In order to stay maintainable ``gtpyc`` will not
add any new logic beyond reading input Python code and command line options.

However, the final version will rely on more fine grained control over when and where backends
create and store intermediate source files, which should become part of the backend API to be used
for run-time compilation in order to avoid redundancy and guarantee maintainability.

Other language projects
+++++++++++++++++++++++

In order to benefit from the higher abstraction level of the GT4Py eDSL it should not be required
to run Python code at runtime. Especially for existing programs written in other languages it makes
more sense to link to libraries created by GT4Py as part of the build process of the host program.

Avoiding runtime dependency
+++++++++++++++++++++++++++

Even for Python projects it may be desirable to distribute only the extension modules created by
GT4Py, not the code that generated them, thus requiring the final user of the generated code to
only install GridTools, not GT4Py.

Licensing
+++++++++

Using only the GT4Py generated stencils in a project without depending on GT4Py at runtime allows
to use a licence other than GPL3 in said project without the express permission of CSCS.

Flexibility
+++++++++++

The import mechanism will allow the flexibility to define GTScript objects in GTScript files and
using them in Python code without extra steps (as if they were defined directly in Python), yet
also compiling them into other language sources / bindings from the same code base just by running
the CLI tool on them. This allows prototyping in Python without making a final choice as to project
language and license.

Usage and Impact
----------------

Basic CLI usage
+++++++++++++++

The usage is explained best using a small example.

Note that ``.gt.py`` files could be replaced by equivalent ``.py`` files (importing GTScript symbols
either from :py:mod:`gt4py` or from a ``.gt.py`` file) in all following examples.  Python modules or packages
are also valid input files to ``gtpyc``, provided they are valid Python under the assumption that the
import extensions are installed.

Assume the following file structure:

.. code-block:: bash

   $ tree .
   pwd
   ├── constants.py
   └── stencils.gt.py

``stencils.gt.py`` contains the GTScript code to be compiled to stencils. The contents might look
something like the following example.

.. code-block:: python
   :caption: stencils.gt.py

   # [GT] using-dsl: gtscript

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

Notice that this file uses names from :py:mod:`gt4py.gtscript` without importing :py:mod:`gt4py`. The names will be
injected by ``gtpyc`` upon recognizing the :code:`# [GT] using-dsl: gtscript` comment.  Also note that
:code:`stencil_b` uses an external value which is not available in the file itself, so it will have to be
supplied on the command line.  The file ``constants.py`` contains some constant values (which might
be templated by the build system).

In order to get C++ code we can now run ``gtpyc`` with for example the GridTools multi core backend
(``-b gtmc``) and tell it to generate the stencils in the new subdirectory ``stencils`` (:code:`-o
stencils`). 

.. code-block:: bash

   $ gtpyc -b gtmc stencils.gt.py -o stencils -e COMPILE_TIME_VALUE 
   $ tree .stencils/
   stencils
   ├── stencil_a.cpp
   ├── stencil_a.hpp
   ├── stencil_b.cpp
   └── stencil_b.hpp

The current backends of :py:mod:`gt4py` (with the exception of the Python-only ones) all have the ability
to generate Python bindings.  Future backends might allow bindings for other languages. This is
accessible through an additional CLI option, which should be validated based on the chosen backend.

.. code-block:: bash

   $ gtpyc -b gtx86 stencils.gt.py -o stencils --bindings=python -e COMPILE_TIME_VALUE 
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

Finally, the backend may allow options specific to it. These can be passed using the :code:`--option` or
:code:`-O` flag.  For example the GridTools multi core backend takes a :code:`debug` flag (which does nothing
during source file generation) but would activate debug flags if we ask gt4py to compile a readily
importable Python extension.

.. code-block:: bash

   $ gtpyc -b gtmc stencils.gt.py -o stencils -e COMPILE_TIME_VALUE -O debug True --bindings=python --compile-bindings
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

Additional Commandline options will mostly correspond to the keyword arguments of the
:py:mod:`gtscript.stencil` decorator.

This should be easy to incorporate into existing build systems as an additional step from ``.py``
source files to ``.cpp`` or ``.cu`` sources before building and linking or as an alternative step to
build ``.py`` sources into ready to link libraries.

Advanced CLI usage
++++++++++++++++++

For complex or mixed language usecases it might be desirable to use a whole library of GTScript /
Python files. The import mechanism makes it possible.

.. code-block:: bash

   $ tree .
   pwd
   ├── stencils.gt.py
   └── lib
       ├── __init__.py
       ├── foo.gt.py
       └── bar
           ├── __init__.py
           └── baz.gt.py

Note that packages require an __init__.py which remains a valid Python module (no :py:mod:`gt4py.gtscript`
injection). However any Python module inside the package can import from any GTScript file
(including :py:mod:`gt4py.gtscript` members).

.. code-block:: bash

   $ gtpyc -b <backend> stencils.gt.py -o stencils

Compiles all top-level stencil members of ``stencils.gt.py``, whether they are defined directly in
``stencils`` or imported from ``lib``

.. code-block:: bash

   $ gtpyc -b <backend> lib -o lib_stencils

Compiles all top-level stencil members of ``lib/__init__.py``.

Usage from Python
+++++++++++++++++

After adding the following to the top of a Python module, any GTScript files in the PYTHONPATH can
be imported as Python modules:

.. code-block:: python

   from gt4py.cartesian import gtpy_import; gtpy_import.install()

Backward compatibility
----------------------

This GDP is aimed to be fully backward-compatible.


Detailed description
--------------------

Any description of design ideas and implementation refers to the `reference implementation
<reference_impl_pr>`_.  This section will be updated as the reference implementation progresses.

Naming
++++++

The accepted name, used throughout this document is ``gtpyc`` which derives from :py:mod:`gt4py` but is easier on
typing.  The ``c`` at the end stands for "compiler". The author does not have a strong prefernce for
this name, it is simply the first one that came to mind.

The accepted conventional file extension for GTScript files is ``.gt.py``. The extension ``.gtpy``
is also allowed for cases where double extensions may not be practical.

Alternatives under consideration:

* ``gtscript`` / ``gtscriptc`` (or short version ``gts`` / ``gtsc``)  -> most intuitive file extension:
  ``.gts`` * same as above but prefixed with ``py`` -> most intuitive file extension: ``.pygt`` or
  ``.pyg``
   
Rejected Alternatives:

* ``gt4pyc``, the sequence "gt4" is all typed with the left index finger on a standard keyboard. The
  author strongly feels that cli command names should start with an easy to type sequence
  (afterwards tab-completion can be used).

It is recommended to allow one file extension for GTScript files which can be derived from the CLI
command name by shortening it in an intuitive way. Since the accepted double extension might cause
trouble for some tools or in some environments an additional fallback is acceptable. It is possible
to allow many more extensions, however the potential confusion outweighs the benefits of being more
permissive.

Enabling all of GTScript without importing from gt4py.cartesian
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The currently chosen route for this is to require a comment at the very start of the file::

   # [GT] using-dsl: gtscript

This will serve two purposes, first it will mark the file as being written in GTScript.  Any name
that in Python can be accessed by :code:`from gt4py.cartesian.gtscript import *` will work when compiling with
``gtpyc`` but will be deemed undefined by the Python interpreter.  It is not planned to provide any
means of informing Python syntax checkers to consider these names as defined.  Secondly ``gtpyc`` can
replace this line with an actual :code:`import` line without changing line numbers for error messages.

Obviously, some symbols like the :code:`@stencil` decorator will have to be either changed or an
alternative has to be offered, since we do not want loading of the input GTScript file to already
trigger a compilation and though we might want to give default arguments to the backend in the
decorator we want to be able to override them on the CLI.

Lazy stencil decorator
++++++++++++++++++++++

The :py:func:`gt4py.gtscript.stencil` decorator will be extended to return an intermediate object, a
drop-in replacement for the compiled :py:class:`StencilObject` which triggers the compilation
process only when used in a way that requires the stencil to be compiled first.  On the other hand
it will hold all contextual information given to the decorator, which will allow ``gtpyc`` to
trigger it's slightly modified build process.

GTScript import system
++++++++++++++++++++++

GTScript files can import Python modules and vice versa, after installing the GTScript import
system (which can be done in a single line). ``gtpyc`` installs the import system and (by default)
adds the parent directory of the input file to :py:mod:`sys.path`, the search path for Python imports. This
means Python and GTScript modules and packages in the same folder as the input file are found by
default, other than that imports behave as normal.  

The public API consists of the :py:mod:`gt4py.gtpy_import.install` function.

Passing externals
+++++++++++++++++

There are two supported ways to configure values at compile / generate time.

* By relative import of a Python file, which may be automatically generated from a template.  The
  latter could happen as part of a build system depending on build parameters. In this case the
  stencil definition can use the values without importing them from :code:`__externals__`. If it does,
  however, the external value can be overriden on the command line using the following second
  option.

* By passing externals options on the command line. In this case the external will be passed to
  every stencil in this run of ``gtpyc`` and each stencil needs to import it from :code:`__externals__` to
  use it.

Generating Language bindings
++++++++++++++++++++++++++++

The intention of this GDP is to support generating language bindings for all languages the chosen
backend supports. These language bindings are intended to be usable without :py:mod:`gt4py` as a
requirement. This is important to allow usage of generated bindings in non-GPL3 projects.

Implications for Tools (IDEs, Linters, etc)
+++++++++++++++++++++++++++++++++++++++++++

It has been remarked that it would be beneficial to use Python tools like linters, checkers, syntax
highlighting etc. for GTScript files.  This should work by default using the recommended ``.gt.py``
file extension. However it is natural that Python tools will flag some code which is perfectly
valid GTScript code as faulty Python code. Most tools should expose configuration options to
ignore or correctly consider such cases.  These configuration options are very different from tool
to tool and are documented for each tool separately. This GDP does not propose packaging any such
configuration or even extensions for tools with :py:mod:`gt4py`.

Note that the following is a simple way to get most of the desired behaviour from any tools which
have trouble with the ``.gt.py`` double extension (The author is not aware of any):

.. code-block:: bash

   $ tree .
   pwd
   ├── mystencils.py
   └── mygts.gt.py

.. code-block:: python
   :caption: mygts.gt.py

   # [GT] using-dsl: gtscript

.. code-block:: python
   :caption: mystencils.py

   from mygts import lazy_stencil, Field, computation, interval

   @lazy_stencil
   def mystencil(a: Field[float]):
      with computation(PARALLEL), interval(...):
         a = 1.

Now IDEs will recognize ``mystencils.py`` as a Python file and will highlight and check the syntax.
Of course tools will be unable to import ``mygts``, unless there is a way to configure them to run
:code:`gt4py.gtpy_import.install()` before trying to import.

Related Work
------------

CLIs of well-known compilers (Provide CLI conventions):

* `clang`_

* `gcc`_

* `gfortran`_

Implementation
--------------

Implementation will start with a proof-of-concept CLI with an absolutely mninimal feature set,
taking a single function in an input ``.py`` file and outputting the result of the stencil
compilation in a separate file.

If it becomes apparent at that stage that changes to the internal structure would become necessary
these will likely be treated in separate GDPs.

The PoC will utilize the `click`_ framework for the CLI, since it encourages separation and reuse
of CLI argument / option handling and documentation code from program logic. None of the known
limitations of `click`_ are foreseen to be detrimental to what this GDP wants to achieve.

Reasons for choosing `click`_
+++++++++++++++++++++++++++++

* separation of concerns

* ease of reuse of CLI components

* built in command completion for bash, zsh etc

* built-in testing api


Alternatives
------------

Using `argparse`_ for the CLI
+++++++++++++++++++++++++++++

Using `argparse`_ has been rejected. although it is not impossible to separate option handling code
from program logic, any attempt to do so consistently would lead to partially reinventing one of
the more advanced frameworks like `click`_.

The author of this GDP does believe the additional requirement of a small pure-Python framework
like `click`_ to be outweighed by the benefits.

Using plain ``.py`` extension in combination with the marker comment
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The author believes that the two types of files serve distinctly separate purposes.  While both
types can be passed into ``gtpyc``, plain ``.py`` files should represent valid Python modules whereas
``.gt.py`` files are treated as written in GTScript, a domain specific language that extends Python.

It may be a subtle difference in implementation but quite a difference in intent. The author of a
``.py`` file may use :py:mod:`gt4py` as a library, whereas the author of a GTScript file uses a different
language which happens to have the same syntax.

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
