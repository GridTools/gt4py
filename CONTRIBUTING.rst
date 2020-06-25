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
always used.

Additionally, general code style should comply with standard style
guidelines for Python programming such as
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__. 

In general, Python modules should be structured in the following order:

1. Shebang line, #! /usr/bin/env python (only for executable scripts)
2. License header (``LICENSE_HEADER.txt``) and module-level comments
3. Module-level docstring
4. ``__all__ = [...]`` statement, if present
5. Imports (alphabetically ordered within each block)

   a. Block of imports from the standard library
   b. Block of imports from general third party libraries (e.g. numpy,
      xarray)
   c. Block of imports from specific submodules of the project

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

Tools
~~~~~

-  Use `Black: the uncoompromising Python code
   formatter <https://github.com/ambv/black>`__ with not more than 120
   characters per source line and 79 for docstrings

-  Follow NumPy format for docstrings with sphinx-Napoleon. Very useful
   guidelines can be found in
   `LSST <https://developer.lsst.io/python/numpydoc.html>`__ docstrings
   conventions

-  Git commit hooks with `pre-commit <https://pre-commit.com/>`__
   - runs formatting and compliance checks for you
   - will be run on all files at every pull request
