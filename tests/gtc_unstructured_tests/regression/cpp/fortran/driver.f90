program main
    use iso_c_binding
    use bindgen_handle
    use vertex2edge_lib
    implicit none
    integer, parameter :: edges = 18, vertices = 9
    real(8), dimension(vertices) :: in
    real(8), dimension(edges) :: out

    in = init()

    !$acc data copyin(in) copyout(out)
    call run_stencil(in, out)
    !$acc end data
    
    if (any(out /= result())) stop 1

    print *, "It works!"

contains
    function init()
        integer, dimension(vertices) :: init
        init = (/    1, 1, 1, &
                     1, 2, 1, &
                     1, 1, 1 /)
    end

    function result()
        integer, dimension(edges) :: result 
        result = (/ 2,  2,  2, &
                    3,  3,  2, &
                    2,  2,  2, &
                  2,  3,  2, &
                  2,  3,  2, &
                  2,  2,  2/)
    end
end
