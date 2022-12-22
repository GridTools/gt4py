============
Contributing
============

Contributions to GT4Py are welcome, and they are greatly appreciated, whether
they be a well-described issue or contributions to the GT4Py code.

Proper credit will be given to contributors by adding their names to the
AUTHORS_ file. ETH Zurich owns the GT4Py library, and external
contributors must sign a contributor assignment agreement.

Bug reports and feature requests
--------------------------------

We always welcome both, don't hesitate opening an issue directly in the `GitHub
repository <https://github.com/GridTools/gt4py/issues>`_. If possible please
add a minimal snippet (or jupyter notebook) of code that illustrates the
problem or feature. The clearer the description the quicker we can get back to
you!

Contributing code
-----------------

We use a [fork and pull request](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) workflow for code contributions.
Pull requests need to pass all the automated checks as well as a review before
they can be merged. You should set up your development environment as
recommended for developers in the README_, and run the automated checks
locally before creating your pull request.

The review will look at difficult to automate things, for example the first
time you contribute you will be asked to sign a legal agreement and to add your
name to AUTHORS_. More things the reviewer will consider are listed below in
`Review Process`_

The reviewer will also be replying to any questions you might have about
requested changes and willing to give advice on how to implement some of
the requests.

The full guidelines are explained in CODING_GUIDELINES_.


.. _README: https://github.com/GridTools/gt4py/blob/main/README.rst
.. _AUTHORS: https://github.com/GridTools/gt4py/blob/main/AUTHORS.rst
.. _CODING_GUIDELINES: https://github.com/GridTools/gt4py/blob/main/CODING_GUIDELINES.rst

Review Process
--------------

Very large pull requests may receive only a partial review at first.
This is to be able to respond within a reasonably short time frame while
still taking time to fairly assess all the changes.

We use CODING_GUIDELINES_ as a reference for reviewing contributions from
a code style and design standpoint.

Further points to be considered are:

- Check all authors are covered by a contributor agreement and added to AUTHORS_

- All added or changed functionality must have a test that fails before and
  passes after changes and documents the correct behaviour.

- The documentation (docstrings as well as ``/docs`` must be kept in sync.

- New files are not allowed to skip ``pre-commit`` checks. Look for additions
  to exclude lists in  pre-commit-config_. The only exception is that
  test modules might skip ``mypy`` checks.

- If a file currently excluded from checks in pre-commit-config_  is changed
  substantially, the reviewer may request dropping it from the exclude list and
  cleaning up the remaining issues.  The remaining issues must be small
  compared to the size of the PR.


.. _pre-commit-config: https://github.com/GridTools/gt4py/blob/main/.pre-commit-config.yaml
