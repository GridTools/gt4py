============
Contributing
============

Contributions to GT4Py are welcome, and they are greatly appreciated, whether
they be a well-described issue or contributions to the GT4Py code.

Proper credit will be given to contributors by adding their names to the
``AUTHORS.rst`` file. ETH Zurich owns the GT4Py library, and external
contributors must sign a contributor assignment agreement.

Bug reports and feature requests
--------------------------------

We always welcome both, don't hesitate opening an issue directly in the `GitHub
repository <https://github.com/GridTools/gt4py>`_. If possible please add a
minimal snippet (or jupyter notebook) of code that illustrates the problem or
feature. The clearer the description the quicker we can get back to you!

Contributing code
-----------------

We use a fork and pull request workflow for code contributions. If that does
not sound familiar to you, do not worry! A simple search for ``github fork pull
request workflow`` will yield many explanations and tutorials.

Pull requests need to pass some automated checks as well as a review before
they can be merged. If you set up your development environment as recommended
for developers in ``README.rst``, it is easy to run the automated checks
locally before creating your pull request.

The review will look at difficult to automate things, for example the first
time you contribute you will be asked to sign a legal agreement and to add your
name to ``AUTHORS.rst``. Other things the reviewer will look at include:

- the presence of tests that fail before and pass after your changes are
  applied
- clean, readable code
- whether the documentation is still in sync after your changes

The reviewer will also be replying to any questions you might have about
requested changes and willing to give advice on how to implement some of
the requests.

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

- To get the same compliance checks that will be run on GitHub at every commit,
  make sure the ``pre-commit`` python package is installed and then run ``$
  pre-commit install`` from the repo directory. If your development environment
  does not contain a Python 3.6 interpreter, run ``pre-commit install-hooks``
  using a separate Python 3.6 environment.
