! GridTools
!
! Copyright (c) 2014-2019, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

program main
    use iso_c_binding
    use bindgen_handle
    use implementation
    implicit none
    integer, parameter :: i = 9, j = 10, k = 11
    real(8), dimension(i, j, k) :: in, out
    type(c_ptr) in_handle, out_handle

    in = initial()

    in_handle = create_data_store(i, j, k)
    out_handle = create_data_store(i, j, k)

    call copy_to_data_store(in_handle, in(:,1,1))
    call run_copy_stencil(in_handle, out_handle)
    call copy_from_data_store(out_handle, out(:,1,1))

    if (any(in /= initial())) stop 1
    if (any(out /= initial())) stop 1

    call bindgen_release(out_handle)
    call bindgen_release(in_handle)

    print *, "It works!"

contains
    function initial()
        integer :: x
        integer, dimension(i, j, k) :: initial
        initial = reshape((/(x, x = 1, size(initial))/) , shape(initial))
    end
end
