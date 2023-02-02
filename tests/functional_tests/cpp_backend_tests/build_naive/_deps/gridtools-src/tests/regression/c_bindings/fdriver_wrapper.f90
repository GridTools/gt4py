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
    use implementation_wrapper
    implicit none
    integer, parameter :: i = 9, j = 10, k = 11
    real(4), dimension(i, j, k) :: in4, out4
    real(8), dimension(i, j, k) :: in8, out8

    in4 = initial()
    call run_copy_functor(in4, out4)
    if (any(in4 /= initial())) stop 1
    if (any(out4 /= initial())) stop 1

    in8 = initial()
    call run_copy_functor(in8, out8)
    if (any(in8 /= initial())) stop 1
    if (any(out8 /= initial())) stop 1

    print *, "It works!"

contains
    function initial()
        integer :: x
        integer, dimension(i, j, k) :: initial
        initial = reshape((/(x, x = 1, size(initial))/) , shape(initial))
    end
end
