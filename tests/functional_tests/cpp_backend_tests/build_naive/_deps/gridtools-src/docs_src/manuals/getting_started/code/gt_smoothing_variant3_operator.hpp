struct smoothing_function_3 {
    using phi = in_accessor<0>;
    using lap = in_accessor<1, extent<-1, 1, -1, 1>>;
    using out = inout_accessor<2>;

    using param_list = make_param_list<phi, lap, out>;

    constexpr static double alpha = 0.5;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, lower_domain) {
        eval(out()) = eval(phi() - alpha * call<lap_function>::with(eval, lap()));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, upper_domain) {
        eval(out()) = eval(phi());
    }
};
