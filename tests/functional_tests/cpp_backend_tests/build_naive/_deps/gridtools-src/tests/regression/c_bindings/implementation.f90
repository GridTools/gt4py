! This file is generated!
module implementation
use iso_c_binding
implicit none
  interface

    subroutine copy_from_data_store(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      real(c_double), dimension(*) :: arg1
    end subroutine
    subroutine copy_to_data_store(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      real(c_double), dimension(*) :: arg1
    end subroutine
    type(c_ptr) function create_data_store(arg0, arg1, arg2) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
    end function
    subroutine run_copy_stencil(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      type(c_ptr), value :: arg1
    end subroutine

  end interface
contains
end
