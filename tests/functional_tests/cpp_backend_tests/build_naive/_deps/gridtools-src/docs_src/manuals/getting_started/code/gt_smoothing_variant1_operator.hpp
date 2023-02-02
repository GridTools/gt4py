struct smoothing_function_1 {
    using phi = in_accessor<0>;
    using laplap = in_accessor<1>;
    using out = inout_accessor<2>;

    using param_list = make_param_list<phi, laplap, out>;

    constexpr static double alpha = 0.5;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, lower_domain) {
        eval(out()) = eval(phi() - alpha * laplap());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, upper_domain) {
        eval(out()) = eval(phi());
    }
};
