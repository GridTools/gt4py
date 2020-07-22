Commandline
===========

GT4Py installs the `gtpyc` command (introduced in
:doc:`GDPs/gdp-0001-standalone-cli`). Implementation is ongoing.

Usage
-----

List available backends
+++++++++++++++++++++++

.. code-block:: bash

   $ gtpyc --list-backends

               computation    bindings        CLI-enabled
   ---------   -----------    --------        -------------
   <backend>   <lang>         <lang>, <lang>  <Yes | No>
   ...

Lists the currently implemented backends with computation language and possible
language bindings.  The last column informs whether CLI support was implemented
for the backend.
