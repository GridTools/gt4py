program main
    use iso_c_binding
    use bindgen_handle
    use vertex2edge_lib
    implicit none
    integer, parameter :: edges = 18, vertices = 9
    real(8), dimension(vertices) :: in
    real(8), dimension(edges) :: out
    integer, dimension(edges,2) :: neigh_tbl
    type(c_ptr) :: stencil

    neigh_tbl(1,:) = (/1,2/)
    neigh_tbl(2,:) = (/2,3/)
    neigh_tbl(3,:) = (/2,1/)
    neigh_tbl(4,:) = (/4,5/)
    neigh_tbl(5,:) = (/5,6/)
    neigh_tbl(6,:) = (/6,4/)
    neigh_tbl(7,:) = (/7,8/)
    neigh_tbl(8,:) = (/8,9/)
    neigh_tbl(9,:) = (/9,7/)
    neigh_tbl(10,:) = (/1,4/)
    neigh_tbl(11,:) = (/2,5/)
    neigh_tbl(12,:) = (/3,6/)
    neigh_tbl(13,:) = (/4,7/)
    neigh_tbl(14,:) = (/5,8/)
    neigh_tbl(15,:) = (/6,9/)
    neigh_tbl(16,:) = (/7,1/)
    neigh_tbl(17,:) = (/8,2/)
    neigh_tbl(18,:) = (/9,3/)

    in = init()

    !$acc data copyin(in) copyout(out)
    stencil = alloc_stencil(edges, neigh_tbl)
    call run_stencil(stencil, in, out)
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
