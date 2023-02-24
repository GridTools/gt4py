=======================================
GDP 2 — Specializations Near Boundaries
=======================================

:Author: Johann Dahm <johannd@vulcan.com> - Vulcan Climate Modeling
:Status: Draft
:Type: Feature
:Created: 21-04-2020
:Discussion PR: `https://github.com/GridTools/gt4py/pull/24 <discussion_pr>`_
:Implementation: `https://github.com/GridTools/gt4py/pull/234 <impl_pr>`_


Abstract
--------

GT4Py treats the vertical domain specially so that top and bottom boundary conditions can easily be applied.
It also has a limited ability to compute on sub-regions of the horizontal iteration space using the ``domain`` and ``origin`` keyword arguments to stencil calls.
Using these offsets and limits the stencil's iteration relative to fields' origins.
This feature however requires that separate stencils are written for boundaries, limiting the possibilities for backend optimization and adding complexity to the application.

Here we propose a feature to push such regional computation around boundaries into the stencil and in the process add a feature capable of expressing stencil computation on global boundaries of domain-decomposed iterations spaces.
This additionally has the benefit of locating stencil-specific boundary calculation adjacent to the main stencil code, resulting in a friendlier interface and fewer opportunities for errors.

Motivation and Scope
--------------------

Numerical models, especially finite difference methods, commonly have specialized computation near boundaries to facilitate boundary conditions or other grid-specific requirements.
We are developing a weather model on a cubed-sphere multi-block structured grid, and have many stencils that require such specific treatment on and near the structured block boundaries.

We are currently using ``domain`` and ``origin`` keyword arguments to separate stencil calls for special block boundary computation, but doing so leads to inefficient code (multiplies stencil call overhead) that is difficult for model developers to read.
To illustrate the need for such a feature, consider a snippet of the model that computes a variable ``ub`` differently based on location in the grid:

.. code-block::

    @gtscript.stencil()​
    def main_ub(uc: Field, vc: Field, cosa: Field, rsina: Field, ub: Field, dt5: float):​
        with computation(PARALLEL), interval(...):​
            ub = dt5 * (uc[0, -1, 0] + uc - ​(vc[-1, 0, 0] + vc) * cosa) * rsina​

    @gtscript.stencil()​
    def x_edge_ub(ut: Field, ub: Field, dt5: float):
        with computation(PARALLEL), interval(...):​
            ub = dt5 * (ut[0, -1, 0] + ut)​

    @gtscript.stencil()​
    def y_edge_ub(ut: Field, ub: Field, *, dt4: float):
        with computation(PARALLEL), interval(...):
            ub = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])

    # Requires a python function to call stencils on each region
    def ubke(uc, vc, ut, ub, dt5, dt4, grid):​
        domain_y_edge = (grid.ni, 1, grid.nz)
        domain_x_edge = (1, grid.nj, grid.nz)
        main_ub(uc, vc, grid.cosa, grid.rsina, ub, dt5=dt5, ​
                origin=(grid.local_istart, grid.local_jstart, 0),
                domain=(grid.ni, grid.nj, grid.nz))​
        if grid.west_edge:​
            x_edge_ub(ut, ub, dt5=dt5, ​origin=(grid.local_istart, grid.local_jstart, 0), ​domain=domain_x_edge)​
        if grid.south_edge:
            y_edge(ut, ub, dt4=dt4, origin=(grid.local_istart, grid.local_jstart, 0), domain=domain_y_edge)
        if grid.north_edge:
            y_edge_ub(ut, ub, dt4=dt4, origin=(grid.local_istart, grid.local_jend, 0), domain=domain_y_edge)
        if grid.east_edge:
            x_edge_ub(ut, ub, dt5=dt5, origin=(grid.local_iend, grid.local_jstart, 0), domain=domain_x_edge)

The specific feature that we are proposing is the addition of a ``with horizontal()`` context that specifies the horizontal iteration space (over parallel axes) using ``region`` objects.
The arguments of ``region`` object's ``__getitem__`` method use *axis offsets* to define a subregion of the stencil computational domain in the parallel ``I`` and ``J`` axes.

Using this specification, the example is transformed into (for an ``NxN`` processor layout):

.. code-block:: python

    row = (procid / N)
    col = (procid % N)
    istart = I[0] - np_local * col
    jstart = J[0] - np_local * row
    iend = I[-1] + np_global - col * (np_local + 1)
    jend = J[-1] + np_global - row * (np_local + 1)

    @gtscript.stencil()
    def stencil(uc: Field, vc: Field, cosa: Field, rsina: Field, ut: Field, ub: Field, dt5: float, dt4: float):
        from __externals__ import istart, iend, jstart, jend
        with computation(PARALLEL), interval(...):
            ub = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina
            with horizontal(region[istart, :], region[iend, :]):
                ub = dt5 * (ut[0, -1, 0] + ut)
            with horizontal(region[:, jstart], region[:, jend]):
                ub = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])

where ``np_local`` and ``np_global`` are the local and global number of horizontal cells in either direction.
For a ``256x256`` horizontal grid using ``4`` processors in a ``2x2`` grid, ``np_local`` and ``np_global`` would be ``128`` and ``256``, respectively.
GT4Py is given all the information it needs to reason about where a computation occurs, without overspecification.
This information is encoded in the offsets relative to the start or stop of the stencil compute domain when it runs.

This greatly reduces the complexity of the code and consolidates operations on ``ub`` - it is now immediately clear what the stencil is filling into ``ub`` everywhere.


Usage and Impact
----------------

This is an optional feature, but will be the accepted approach to specialize computation at points in the horizontal iteration space.


Backward Compatibility
----------------------

This GDP aims to be fully backward-compatible.


Detailed Description
--------------------

As introduced above, we propose adding a new ``with horizontal()`` context that specializes the stencil on a region of the horizontal axes bounds using ``region`` objects, which pass information to GT4Py about the horizontal iteration space through the indexing operator, similar to numpy arrays.


Axis Offsets
++++++++++++

Regions computation is specified using `Axis Offsets`, which are defined in GT4Py by subscripting the axes (``I``, ``J``, and ``K``).
These may be indexed and returns the specific indices within a stencil relative to the compute origin.
For example: ``I[0]`` is the first compute point, ``I[1]`` the second, and finally ``I[-1]`` is the last point in the stencil compute domain along the ``I`` axis.

Stencil computation in the horizontal axes behaves differently than in the vertical because statements execute over an index space that may extend beyond the limits defined in the stencil compute domain.
Such ``extents`` cannot be represented by merely subscripting axes, since for example ``I[-1]`` referes to the last compute domain index along the ``I`` axis, not the point before the beginning of it.
Axis Offsets therefore internally hold an offset which is added or subtracted from the indexed point in the axis.
For example ``I[0] - 2`` is itself an Axis Offset that refers to 2 points before the start of the compute domain in ``I``.

Axis Offsets may be manipulated in Python or in a stencil and can be used as externals in GT4Py to be used in ``region`` subscripts.

Region Specification
++++++++++++++++++++

``region`` is a keyword in GT4Py that, when subscripted using slices of axis offsets, defines the restricted computation.
These form the arguments to ``with horizontal()``.

As an example, ``region[I[0], :]`` specifies a restriction to the first compute point on the ``I`` axis, and no restriction in the ``J`` axis.
``I[0]`` is a single point, so when converted to a slice is still the single point.
The ``J`` axis simply has ``:``, which is an unrestricted slice, which GT4Py interprets as an unrestricted axis (behaves normally).

The previous example introduced a key element of regional computation: There must not only be a way of specifying axis offsets outside the compute domain, but slices that extend to infinity in each direction (or alternatively, unrestricted endpoints of axes).
GTScript interprets an unrestricted start or stop element as extending to infinity (or, unrestricted).
This is useful in the case when writing a stencil and requiring that an edge condition be made without knowing how far the statements needs to be extended.
For example:

.. code-block:: python

    with computation(PARALLEL), interval(...):
        with horizontal(region[:I[0], :]):
            u = 0

This will set ``u=0`` in all extended computation points to the left of the compute domain.

Examples of this are shown in the image below.
The blue line shows the compute domain along the ``I`` axis, and two examples of region axis slices are shown in red.

.. image:: _static/gdp-0002/axis_offsets.svg

Execution
+++++++++

Another key feature to remember when using regions is that these should be thought of as specifying specialized computation at points.
These are therefore not guaranteed to execute, except where inside the compute domain.
The statements inside a block with ``region[:I[0]-1, :]`` will only execute where the outputs from that block are necessary to compute something else with an extent.
For example, the following will execute

.. code:: python

    with computation(PARALLEL), interval(...):
        with horizontal(region[I[0]-1, :]):
            field_in = 0.0
        field_out = field_in[-1, 0, 0] + field_in[0, 0, 0]

since the ``field_in`` value at ``I[0]-1`` is being consumed to compute a value of an output field inside the compute domain.
If the region were defined using ``I[0]-2``, the code would be ignored.


Implementation
--------------

The implementation in GT4Py involves

1. Correctly parse ``with horizontal()`` in the frontend, and add ability for IRs to represent this computation
2. Add parsing tests
3. Add code generation support
4. Code generation tests
5. Create at least one demo that incorporates this feature


FV3 Example
-----------

.. code-block:: Fortran

    subroutine divergence_corner(u, v, ua, va, divg_d, ...)

    ! arguments
    real :: ua(isd:ied, jsd:jed)          ! cell-center
    real :: va(isd:ied, jsd:jed)          ! cell-center
    real :: u(isd:ied, jsd:jed+1)         ! staggered in y-direction
    real :: v(isd:ied+1, jsd:jed)         ! staggered in x-direction
    real :: divg_d(isd:ied+1, jsd:jed+1)  ! corner (staggered both in x- and y-direction)

    ! locals
    real :: uf(is-2:ie+2, js-1:je+2)      ! staggered in y-direction
    real :: vf(is-1:ie+2, js-2:je+2)      ! staggered in y-direction

    ! indices
    integer :: is,  ie,  js,  je   ! compute domain
    integer :: isd, ied, jsd, jed  ! data domain = compute domain + halo zone

    is2 = max(2, is)         ! restrict computation to exclude west-edge
    ie1 = min(npx-1, ie+1)   ! restrict computation to exclude east-edge

    do j = js, je+1
      if (j == 1 .or. j == npy) then
        do i = is-1, ie+1
          uf(i,j) =
            u(i,j)*dyc(i,j)*0.5*(sin_sg(i,j-1,4) + sin_sg(i,j,2))
        end do
      else
        do i = is-1, ie+1
          uf(i,j) = &
              (u(i,j) - 0.25*(va(i,j-1) + va(i,j))*(cos_sg(i,j-1,4) + cos_sg(i,j,2)))  &
                                      *dyc(i,j)*0.5*(sin_sg(i,j-1,4) + sin_sg(i,j,2))
        end do
      end if
    end do

    do j = js-1, je+1
      do i = is2, ie1     ! inner domain (full compute domain for ranks without edges)
        vf(i, j) = &
          (v(i,j) - 0.25*(ua(i-1,j) + ua(i, j))*(cos_sg(i-1,j,3) + cos_sg(i,j,1)))  &
                                *dxc(i,j)*0.5*(sin_sg(i-1,j,3) + sin_sg(i,j,1))
      end do
      if (is == 1) &      ! west-edge
        vf(1, j) = &
          v(1, j)*dxc(1, j)*0.5*(sin_sg(0, j, 3) + sin_sg(1, j, 1))
      if (ie+1 == npx) &  ! east-edge
        vf(npx, j) = &
          v(npx, j)*dxc(npx,j)*0.5*(sin_sg(npx-1, j, 3) + sin_sg(npx, j, 1))
    end do

    do j=js,je+1
      do i=is,ie+1
        divg_d(i,j) = vf(i,j-1) - vf(i,j) + uf(i-1,j) - uf(i,j)
      end do
    end do

    if (gridstruct%sw_corner) &
      divg_d(1,    1) = divg_d(1,    1) - vf(1,    0)
    if (gridstruct%se_corner) &
      divg_d(npx,  1) = divg_d(npx,  1) - vf(npx,  0)
    if (gridstruct%ne_corner) &
      divg_d(npx,npy) = divg_d(npx,npy) + vf(npx,npy)
    if (gridstruct%nw_corner) &
      divg_d(1,  npy) = divg_d(1,  npy) + vf(1,  npy)

    do j=js,je+1
      do i=is,ie+1
        divg_d(i,j) = rarea_c(i,j) * divg_d(i,j)
      end do
    end do

.. code-block:: python

    row = (procid / N)
    col = (procid % N)
    istart = I[0] - np_local * col
    jstart = J[0] - np_local * row
    iend = I[-1] + np_global - col * (np_local + 1)
    jend = J[-1] + np_global - row * (np_local + 1)

    @gtscript.stencil(...)
    def divergence_corner(...):
        from __externals__ import istart, iend, jstart, jend
        with computation(PARALLEL), interval(...):
            uf = (u - 0.25*(va[0, -1, 0] + va)*(cos_sg4[0, -1, 0] + cos_sg2))  \
                                      *dyc*0.5*(sin_sg4[0, -1, 0] + sin_sg2)
            with horizontal(region[:, jstart], region[:, jend)):
                uf = u*dyc*0.5*(sin_sg4[0, -1, 0] + sin_sg2)

            vf = (v - 0.25*(ua[-1, 0, 0] + ua)*(cos_sg3[-1, 0, 0] + cos_sg1))  \
                                      *dxc*0.5*(sin_sg3[-1, 0, 0] + sin_sg1)
            with horizontal(region[istart, :], region[iend, :]):
                vf = v*dxc*0.5*(sin_sg3[-1, 0, 0] + sin_sg1)

            divg_d = rarea_c * (vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf)
            with horizontal(region[istart, jstart], region[istart, jend]):
                divg_d = rarea_c * (-vf[0, 0, 0] + uf[-1, 0, 0] - uf)
            with horizontal(region[iend, jstart], region[iend, jend]):
                divg_d = rarea_c * (vf[0, -1, 0] + uf[-1, 0, 0] - uf)


Copyright
---------

This document has been placed in the public domain.
