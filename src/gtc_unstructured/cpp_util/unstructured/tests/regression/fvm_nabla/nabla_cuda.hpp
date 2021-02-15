#pragma once

#ifndef __CUDACC__
#error "Tried to compile CUDA code with a regular C++ compiler."
#endif

#include <gridtools/next/mesh.hpp>
#include <gridtools/next/tmp_storage.hpp>
#include <gridtools/sid/allocator.hpp>

struct connectivity_tag;
struct S_MXX_tag;
struct S_MYY_tag;
struct zavgS_MXX_tag;
struct zavgS_MYY_tag;
struct zavg_tag;

struct pnabla_MXX_tag;
struct pnabla_MYY_tag;
struct vol_tag;
struct sign_tag;
struct pp_tag;

template <class ConnInfoE2V, class EdgePtrs, class EdgeStrides, class VertexNeighborPtrs, class VertexNeighborStrides>
__global__ void nabla_edge_1(ConnInfoE2V e2v,
    EdgePtrs edge_ptr_holders,
    EdgeStrides edge_strides,
    VertexNeighborPtrs vertex_neighbor_ptr_holders,
    VertexNeighborStrides vertex_neighbor_strides) {
    { // first edge loop (this is the fused version without temporary)
        // ===
        //   for (auto const &t : getEdges(LibTag{}, mesh)) {
        //     double zavg =
        //         (double)0.5 *
        //         (m_sparse_dimension_idx = 0,
        //          reduceVertexToEdge(mesh, t, (double)0.0,
        //                             [&](auto &lhs, auto const &redIdx) {
        //                               lhs += pp(deref(LibTag{}, redIdx), k);
        //                               m_sparse_dimension_idx++;
        //                               return lhs;
        //                             }));
        //     zavgS_MXX(deref(LibTag{}, t), k) =
        //         S_MXX(deref(LibTag{}, t), k) * zavg;
        //     zavgS_MYY(deref(LibTag{}, t), k) =
        //         S_MYY(deref(LibTag{}, t), k) * zavg;
        //   }
        // ===
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= e2v.size)
            return;

        auto edge_ptrs = edge_ptr_holders();

        gridtools::sid::shift(edge_ptrs, gridtools::device::at_key<edge>(edge_strides), idx);

        double acc = 0.;
        { // reduce
            for (int neigh = 0; neigh < e2v.max_neighbors; ++neigh) {
                // body
                auto absolute_neigh_index = *gridtools::device::at_key<connectivity_tag>(edge_ptrs);
                auto vertex_ptrs = vertex_neighbor_ptr_holders();
                gridtools::sid::shift(
                    vertex_ptrs, gridtools::device::at_key<vertex>(vertex_neighbor_strides), absolute_neigh_index);

                acc += *gridtools::device::at_key<pp_tag>(vertex_ptrs);
                // body end

                gridtools::sid::shift(edge_ptrs, gridtools::device::at_key<neighbor>(edge_strides), 1);
            }
            gridtools::sid::shift(edge_ptrs, gridtools::device::at_key<neighbor>(edge_strides), -e2v.max_neighbors);
        }
        *gridtools::device::at_key<zavg_tag>(edge_ptrs) =
            0.5 * acc; // via temporary for non-optimized parallel model
        *gridtools::device::at_key<zavgS_MXX_tag>(edge_ptrs) =
            *gridtools::device::at_key<S_MXX_tag>(edge_ptrs) * *gridtools::device::at_key<zavg_tag>(edge_ptrs);
        *gridtools::device::at_key<zavgS_MYY_tag>(edge_ptrs) =
            *gridtools::device::at_key<S_MYY_tag>(edge_ptrs) * *gridtools::device::at_key<zavg_tag>(edge_ptrs);
    }
}

template <class ConnInfoV2E,
    class VertexOrigins,
    class VertexStrides,
    class EdgeNeighborOrigins,
    class EdgeNeighborStrides>
__global__ void nabla_vertex_2(ConnInfoV2E v2e,
    VertexOrigins vertex_origins,
    VertexStrides vertex_strides,
    EdgeNeighborOrigins edge_neighbor_origins,
    EdgeNeighborStrides edge_neighbor_strides) {
    // vertex loop
    // for (auto const &t : getVertices(LibTag{}, mesh)) {
    //     pnabla_MXX(deref(LibTag{}, t), k) =
    //         (m_sparse_dimension_idx = 0,
    //          reduceEdgeToVertex(
    //              mesh, t, (double)0.0, [&](auto &lhs, auto const &redIdx) {
    //                lhs += zavgS_MXX(deref(LibTag{}, redIdx), k) *
    //                       sign(deref(LibTag{}, t), m_sparse_dimension_idx,
    //                       k);
    //                m_sparse_dimension_idx++;
    //                return lhs;
    //              }));
    //   }
    //   for (auto const &t : getVertices(LibTag{}, mesh)) {
    //     pnabla_MYY(deref(LibTag{}, t), k) =
    //         (m_sparse_dimension_idx = 0,
    //          reduceEdgeToVertex(
    //              mesh, t, (double)0.0, [&](auto &lhs, auto const &redIdx) {
    //                lhs += zavgS_MYY(deref(LibTag{}, redIdx), k) *
    //                       sign(deref(LibTag{}, t), m_sparse_dimension_idx,
    //                       k);
    //                m_sparse_dimension_idx++;
    //                return lhs;
    //              }));
    //   }

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= v2e.size)
        return;

    auto vertex_ptrs = vertex_origins();

    gridtools::sid::shift(vertex_ptrs, gridtools::device::at_key<vertex>(vertex_strides), idx);

    *gridtools::device::at_key<pnabla_MXX_tag>(vertex_ptrs) = 0.;
    { // reduce
        for (int neigh = 0; neigh < v2e.max_neighbors; ++neigh) {
            // body
            auto absolute_neigh_index = *gridtools::device::at_key<connectivity_tag>(vertex_ptrs);
            if (absolute_neigh_index != v2e.skip_value) {
                auto edge_ptrs = edge_neighbor_origins();
                gridtools::sid::shift(
                    edge_ptrs, gridtools::device::at_key<edge>(edge_neighbor_strides), absolute_neigh_index);

                auto zavgS_MXX_value = *gridtools::device::at_key<zavgS_MXX_tag>(edge_ptrs);
                auto sign_value = *gridtools::device::at_key<sign_tag>(vertex_ptrs);

                *gridtools::device::at_key<pnabla_MXX_tag>(vertex_ptrs) += zavgS_MXX_value * sign_value;
                // body end
            }
            gridtools::sid::shift(vertex_ptrs, gridtools::device::at_key<neighbor>(vertex_strides), 1);
        }
        gridtools::sid::shift(vertex_ptrs,
            gridtools::device::at_key<neighbor>(vertex_strides),
            -v2e.max_neighbors); // or reset ptr to origin and shift ?
    }
    *gridtools::device::at_key<pnabla_MYY_tag>(vertex_ptrs) = 0.;
    { // reduce
        for (int neigh = 0; neigh < v2e.max_neighbors; ++neigh) {
            // body
            auto absolute_neigh_index = *gridtools::device::at_key<connectivity_tag>(vertex_ptrs);
            if (absolute_neigh_index != v2e.skip_value) {
                auto edge_ptrs = edge_neighbor_origins();
                gridtools::sid::shift(
                    edge_ptrs, gridtools::device::at_key<edge>(edge_neighbor_strides), absolute_neigh_index);

                auto zavgS_MYY_value = *gridtools::device::at_key<zavgS_MYY_tag>(edge_ptrs);
                auto sign_value = *gridtools::device::at_key<sign_tag>(vertex_ptrs);
                ;

                *gridtools::device::at_key<pnabla_MYY_tag>(vertex_ptrs) += zavgS_MYY_value * sign_value;
                // body end
            }
            gridtools::sid::shift(vertex_ptrs, gridtools::device::at_key<neighbor>(vertex_strides), 1);
        }
        gridtools::sid::shift(vertex_ptrs, gridtools::device::at_key<neighbor>(vertex_strides), -v2e.max_neighbors);
    }
}
// ===
//   do jedge = 1,dstruct%nb_pole_edges
//     iedge = dstruct%pole_edges(jedge)
//     ip2   = dstruct%edges(2,iedge)
//     ! correct for wrong Y-derivatives in previous loop
//     pnabla(MYY,ip2) = pnabla(MYY,ip2)+2.0_wp*zavgS(MYY,iedge)
//   end do
// ===
//   {
//     auto pe2v = gridtools::next::mesh::connectivity<
//         std::tuple<atlas::pole_edge, vertex>>(mesh);
//     for (int i = 0; i < gridtools::next::connectivity::size(pe2v);
//          ++i) {
//     }
//   }
template <class VertexOrigins, class VertexStrides>
__global__ void nabla_vertex_4(std::size_t size, VertexOrigins vertex_origins, VertexStrides vertex_strides) {
    // vertex loop
    // for (auto const &t : getVertices(LibTag{}, mesh)) {
    //     pnabla_MXX(deref(LibTag{}, t), k) =
    //         pnabla_MXX(deref(LibTag{}, t), k) / vol(deref(LibTag{}, t), k);
    //     pnabla_MYY(deref(LibTag{}, t), k) =
    //         pnabla_MYY(deref(LibTag{}, t), k) / vol(deref(LibTag{}, t), k);
    //   }

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    auto vertex_ptrs = vertex_origins();

    gridtools::sid::shift(vertex_ptrs, gridtools::device::at_key<vertex>(vertex_strides), idx);

    *gridtools::device::at_key<pnabla_MXX_tag>(vertex_ptrs) /= *gridtools::device::at_key<vol_tag>(vertex_ptrs);
    *gridtools::device::at_key<pnabla_MYY_tag>(vertex_ptrs) /= *gridtools::device::at_key<vol_tag>(vertex_ptrs);
}

std::tuple<int, int> cuda_setup(int N) {
    int threads_per_block = 32;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    return {blocks, threads_per_block};
}

template <class Mesh,
    class S_MXX_t,
    class S_MYY_t,
    class pp_t,
    class pnabla_MXX_t,
    class pnabla_MYY_t,
    class vol_t,
    class sign_t>
void nabla(Mesh &&mesh,
    S_MXX_t &&S_MXX,
    S_MYY_t &&S_MYY,
    pp_t &&pp,
    pnabla_MXX_t &&pnabla_MXX,
    pnabla_MYY_t &&pnabla_MYY,
    vol_t &&vol,
    sign_t &&sign) {
    namespace tu = gridtools::tuple_util;
    // allocate temporary field storage
    int k_size = 1; // TODO
    auto cuda_alloc =
        gridtools::sid::device::make_cached_allocator(&gridtools::cuda_util::cuda_malloc<char[]>); // TODO
    auto zavgS_MXX = gridtools::next::make_simple_tmp_storage<edge, double>(
        (int)gridtools::next::connectivity::size(gridtools::next::mesh::connectivity<edge>(mesh)), k_size, cuda_alloc);
    auto zavgS_MYY = gridtools::next::make_simple_tmp_storage<edge, double>(
        (int)gridtools::next::connectivity::size(gridtools::next::mesh::connectivity<edge>(mesh)), k_size, cuda_alloc);
    {
        auto e2v = gridtools::next::mesh::connectivity<std::tuple<edge, vertex>>(mesh);
        static_assert(gridtools::is_sid<decltype(gridtools::next::connectivity::neighbor_table(e2v))>{});

        auto zavg = gridtools::next::make_simple_tmp_storage<edge, double>(
            (int)gridtools::next::connectivity::size(gridtools::next::mesh::connectivity<edge>(mesh)), k_size, cuda_alloc);

        auto edge_fields = tu::make<gridtools::sid::composite::
                keys<connectivity_tag, S_MXX_tag, S_MYY_tag, zavgS_MXX_tag, zavgS_MYY_tag, zavg_tag>::values>(
            gridtools::next::connectivity::neighbor_table(e2v), S_MXX, S_MYY, zavgS_MXX, zavgS_MYY, zavg);
        static_assert(gridtools::sid::concept_impl_::is_sid<decltype(edge_fields)>{});

        auto edge_ptrs = gridtools::sid::get_origin(edge_fields);
        auto edge_strides = gridtools::sid::get_strides(edge_fields);

        auto vertex_neighbor_fields = tu::make<gridtools::sid::composite::keys<pp_tag>::values>(pp);
        auto vertex_neighbor_ptrs = gridtools::sid::get_origin(vertex_neighbor_fields);
        auto vertex_neighbor_strides = gridtools::sid::get_strides(vertex_neighbor_fields);

        auto [blocks, threads_per_block] = cuda_setup(gridtools::next::connectivity::size(e2v));

        auto e2v_info = gridtools::next::connectivity::extract_info(e2v);

        nabla_edge_1<<<blocks, threads_per_block>>>(
            e2v_info, edge_ptrs, edge_strides, vertex_neighbor_ptrs, vertex_neighbor_strides);
        GT_CUDA_CHECK(cudaDeviceSynchronize());
    } // namespace gridtools::tuple_util;
    {
        auto v2e = gridtools::next::mesh::connectivity<std::tuple<vertex, edge>>(mesh);
        static_assert(gridtools::is_sid<decltype(gridtools::next::connectivity::neighbor_table(v2e))>{});

        auto vertex_fields = tu::make<gridtools::sid::composite::
                keys<connectivity_tag, pnabla_MXX_tag, pnabla_MYY_tag, sign_tag, vol_tag>::values>(
            gridtools::next::connectivity::neighbor_table(v2e), pnabla_MXX, pnabla_MYY, sign, vol);
        static_assert(gridtools::sid::concept_impl_::is_sid<decltype(vertex_fields)>{});

        auto edge_neighbor_fields =
            tu::make<gridtools::sid::composite::keys<zavgS_MXX_tag, zavgS_MYY_tag>::values>(zavgS_MXX, zavgS_MYY);

        auto v2e_info = gridtools::next::connectivity::extract_info(v2e);

        auto [blocks, threads_per_block] = cuda_setup(gridtools::next::connectivity::size(v2e));
        nabla_vertex_2<<<blocks, threads_per_block>>>(v2e_info,
            gridtools::sid::get_origin(vertex_fields),
            gridtools::sid::get_strides(vertex_fields),
            gridtools::sid::get_origin(edge_neighbor_fields),
            gridtools::sid::get_strides(edge_neighbor_fields));
        GT_CUDA_CHECK(cudaDeviceSynchronize());
    }
    {
        auto vertices = gridtools::next::mesh::connectivity<std::tuple<vertex>>(mesh);

        auto vertex_fields = tu::make<gridtools::sid::composite::keys<pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>::values>(
            pnabla_MXX, pnabla_MYY, vol);
        static_assert(gridtools::sid::concept_impl_::is_sid<decltype(vertex_fields)>{});

        auto size = gridtools::next::connectivity::size(vertices);

        auto [blocks, threads_per_block] = cuda_setup(gridtools::next::connectivity::size(vertices));
        nabla_vertex_4<<<blocks, threads_per_block>>>(
            size, gridtools::sid::get_origin(vertex_fields), gridtools::sid::get_strides(vertex_fields));
        GT_CUDA_CHECK(cudaDeviceSynchronize());
    }
}
