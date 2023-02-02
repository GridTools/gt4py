auto phi = make_storage();
auto phi_new = make_storage();
auto lap = make_storage();
auto laplap = make_storage();

halo_descriptor boundary_i(halo, halo, halo, Ni - halo - 1, Ni);
halo_descriptor boundary_j(halo, halo, halo, Nj - halo - 1, Nj);
auto grid = make_grid(boundary_i, boundary_j, axis_t(kmax, Nk - kmax));

auto const spec = [](auto phi, auto phi_new, auto lap, auto laplap) {
    return execute_parallel()
        .stage(lap_function(), phi, lap)
        .stage(lap_function(), lap, laplap)
        .stage(smoothing_function_1(), phi, laplap, phi_new);
};

run(spec, stencil_backend_t(), grid, phi, phi_new, lap, laplap);
