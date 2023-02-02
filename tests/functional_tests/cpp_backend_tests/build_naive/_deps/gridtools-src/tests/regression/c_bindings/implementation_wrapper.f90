! This file is generated!
module implementation_wrapper
use iso_c_binding
implicit none
  interface

    subroutine run_copy_functor0_impl(arg0, arg1) bind(c, name="run_copy_functor0")
      use iso_c_binding
      use bindgen_array_descriptor
      type(bindgen_fortran_array_descriptor) :: arg0
      type(bindgen_fortran_array_descriptor) :: arg1
    end subroutine
    subroutine run_copy_functor1_impl(arg0, arg1) bind(c, name="run_copy_functor1")
      use iso_c_binding
      use bindgen_array_descriptor
      type(bindgen_fortran_array_descriptor) :: arg0
      type(bindgen_fortran_array_descriptor) :: arg1
    end subroutine

  end interface
  interface run_copy_functor
    procedure run_copy_functor0, run_copy_functor1
  end interface
contains
    subroutine run_copy_functor0(arg0, arg1)
      use iso_c_binding
      use bindgen_array_descriptor
      real(c_double), dimension(:,:,:), target :: arg0
      real(c_double), dimension(:,:,:), target :: arg1
      type(bindgen_fortran_array_descriptor) :: descriptor0
      type(bindgen_fortran_array_descriptor) :: descriptor1

      !$acc data present(arg0)
      !$acc host_data use_device(arg0)
      descriptor0%rank = 3
      descriptor0%type = 6
      descriptor0%dims = reshape(shape(arg0), &
        shape(descriptor0%dims), (/0/))
      descriptor0%data = c_loc(arg0(lbound(arg0, 1),lbound(arg0, 2),lbound(arg0, 3)))
      !$acc end host_data
      !$acc end data

      !$acc data present(arg1)
      !$acc host_data use_device(arg1)
      descriptor1%rank = 3
      descriptor1%type = 6
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2),lbound(arg1, 3)))
      !$acc end host_data
      !$acc end data

      call run_copy_functor0_impl(descriptor0, descriptor1)
    end subroutine
    subroutine run_copy_functor1(arg0, arg1)
      use iso_c_binding
      use bindgen_array_descriptor
      real(c_float), dimension(:,:,:), target :: arg0
      real(c_float), dimension(:,:,:), target :: arg1
      type(bindgen_fortran_array_descriptor) :: descriptor0
      type(bindgen_fortran_array_descriptor) :: descriptor1

      !$acc data present(arg0)
      !$acc host_data use_device(arg0)
      descriptor0%rank = 3
      descriptor0%type = 5
      descriptor0%dims = reshape(shape(arg0), &
        shape(descriptor0%dims), (/0/))
      descriptor0%data = c_loc(arg0(lbound(arg0, 1),lbound(arg0, 2),lbound(arg0, 3)))
      !$acc end host_data
      !$acc end data

      !$acc data present(arg1)
      !$acc host_data use_device(arg1)
      descriptor1%rank = 3
      descriptor1%type = 5
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2),lbound(arg1, 3)))
      !$acc end host_data
      !$acc end data

      call run_copy_functor1_impl(descriptor0, descriptor1)
    end subroutine
end
