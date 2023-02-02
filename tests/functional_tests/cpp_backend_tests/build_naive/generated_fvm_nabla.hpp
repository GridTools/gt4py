
    #include <cmath>
    #include <cstdint>
    #include <gridtools/fn/unstructured.hpp>

    namespace generated{

    namespace gtfn = ::gridtools::fn;

    namespace{
    using namespace ::gridtools::literals;

    
                using Vertex_t = gtfn::unstructured::dim::horizontal;
        constexpr inline Vertex_t Vertex{};
        

                using K_t = gtfn::unstructured::dim::vertical;
        constexpr inline K_t K{};
        

            struct V2E_t{};
        constexpr inline V2E_t V2E{};
        

            struct E2V_t{};
        constexpr inline E2V_t E2V{};
        
    
        struct _fun_1 {
            constexpr auto operator()() const {
                return [](auto const& pp,auto const& S_M,auto const& sign,auto const& vol){
                    return [=](auto _cs_1,auto _cs_2){return gtfn::make_tuple((gtfn::tuple_get(0_c,[=](auto _step_3){return _step_3(_step_3(_step_3(_step_3(_step_3(_step_3(_cs_1,0_c),1_c),2_c),3_c),4_c),5_c);}([=](auto _acc_1,auto _i_2){return ((gtfn::can_deref(gtfn::shift(pp,V2E,_i_2))&&gtfn::can_deref(gtfn::shift(S_M,V2E,_i_2)))?[=](auto zavgS,auto sign_){return gtfn::make_tuple((gtfn::tuple_get(0_c,_acc_1)+(gtfn::tuple_get(0_c,zavgS)*sign_)),(gtfn::tuple_get(1_c,_acc_1)+(gtfn::tuple_get(1_c,zavgS)*sign_)));}([=](auto pp){return [=](auto _cs_3){return gtfn::make_tuple((gtfn::tuple_get(0_c,_cs_3)*(0.5*(gtfn::deref(gtfn::shift(pp,E2V,0_c))+gtfn::deref(gtfn::shift(pp,E2V,1_c))))),(gtfn::tuple_get(1_c,_cs_3)*(0.5*(gtfn::deref(gtfn::shift(pp,E2V,0_c))+gtfn::deref(gtfn::shift(pp,E2V,1_c))))));}(gtfn::deref(gtfn::shift(S_M,V2E,_i_2)));}(gtfn::shift(pp,V2E,_i_2)),gtfn::tuple_get(_i_2,gtfn::deref(sign))):_acc_1);}))/_cs_2),(gtfn::tuple_get(1_c,[=](auto _step_6){return _step_6(_step_6(_step_6(_step_6(_step_6(_step_6(_cs_1,0_c),1_c),2_c),3_c),4_c),5_c);}([=](auto _acc_4,auto _i_5){return ((gtfn::can_deref(gtfn::shift(pp,V2E,_i_5))&&gtfn::can_deref(gtfn::shift(S_M,V2E,_i_5)))?[=](auto zavgS,auto sign_){return gtfn::make_tuple((gtfn::tuple_get(0_c,_acc_4)+(gtfn::tuple_get(0_c,zavgS)*sign_)),(gtfn::tuple_get(1_c,_acc_4)+(gtfn::tuple_get(1_c,zavgS)*sign_)));}([=](auto pp){return [=](auto _cs_4){return gtfn::make_tuple((gtfn::tuple_get(0_c,_cs_4)*(0.5*(gtfn::deref(gtfn::shift(pp,E2V,0_c))+gtfn::deref(gtfn::shift(pp,E2V,1_c))))),(gtfn::tuple_get(1_c,_cs_4)*(0.5*(gtfn::deref(gtfn::shift(pp,E2V,0_c))+gtfn::deref(gtfn::shift(pp,E2V,1_c))))));}(gtfn::deref(gtfn::shift(S_M,V2E,_i_5)));}(gtfn::shift(pp,V2E,_i_5)),gtfn::tuple_get(_i_5,gtfn::deref(sign))):_acc_4);}))/_cs_2));}(gtfn::make_tuple(0.0,0.0),gtfn::deref(vol));
                };
            }
        };
    

    inline auto nabla_fencil = [](auto... connectivities__){
        return [connectivities__...](auto backend, auto&& n_vertices,auto&& n_levels,auto&& out,auto&& pp,auto&& S_M,auto&& sign,auto&& vol){
            auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);
            
            
        make_backend(backend, gtfn::unstructured_domain(::gridtools::tuple((n_vertices-0_c),(n_levels-0_c)), ::gridtools::tuple(0_c,0_c), connectivities__...)).stencil_executor()().arg(out).arg(pp).arg(S_M).arg(sign).arg(vol).assign(0_c, _fun_1() , 1_c,2_c,3_c,4_c).execute();
        
        };
    };
    }
    }
    