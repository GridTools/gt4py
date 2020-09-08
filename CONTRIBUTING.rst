============
Contributing
============

Contributions to GT4Py are welcome, and they are greatly appreciated. Proper
credit will be given to contributors by adding their names to the
``AUTHORS.rst`` file. ETH Zurich owns the GT4Py library, and external
contributors must sign a contributor assignment agreement.


Development guidelines
----------------------

Code style
~~~~~~~~~~

`Black <https://github.com/ambv/black>`__ code formatter should be
always used. This is done automatically for you after you set up
`pre-commit` (see below in "Tools" section).

Additionally, general code style should comply with standard style
guidelines for Python programming such as
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__. 

In general, Python modules should be structured in the following order:

1. Shebang line, #! /usr/bin/env python (only for executable scripts)
2. License header (``LICENSE_HEADER.txt``) and module-level comments
3. Module-level docstring
4. ``__all__ = [...]`` statement, if present
5. Imports (sorting and grouping automated by pre-commit hook)
6. Private module variables, functions and classes (names start with
   underscore)
7. Public module variables, functions and classes

General coding advices:

-  ``from X import Y`` import form is generally preferred over
   ``import X``
-  Absolute imports (``from library import something``) SHOULD be
   preferred over relative imports
   (``from .submodule import something``)
-  **is** and **is not** SHOULD be used when comparing to **None**
-  The **set** type SHOULD be used for unordered collections
-  **super()** MAY be used to call parent class methods
-  Iterators and generators SHOULD be used to iterate over large data
   sets efficiently
-  Block comments SHOULD reference the code following them and SHOULD be
   indented to the same level

Also checkout ``REVIEW_CHECKLIST`` to see what code reviewers will look for
beyond these tipps and beyond what is automatically checked for.

Tools
~~~~~

- To get the same compliance checks that will be run on GitHub at every commit,
  make sure the ``pre-commit`` python package is installed and then run ``$
  pre-commit install`` from the repo directory. If your development environment
  does not contain a Python 3.6 interpreter, run ``pre-commit install-hooks``
  using a separate Python 3.6 environment.
