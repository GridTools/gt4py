/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <test_environment.hpp>
#include <timer_select.hpp>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

namespace {
    struct state {
        std::array<int, 3> m_d = {};
        size_t m_steps = 0;
        bool m_needs_verification = true;
        int m_argc;
        char **m_argv;
    } s_state;

    bool init(int argc, char **argv) {
        assert(argc > 0);
        s_state.m_argc = 1;
        s_state.m_argv = argv;

        // Let's assume that our stuff goes the last in the command line.
        // so we can expect some third-party flags at the beginning.
        while (argc > 1 && argv[1][0] == '-') {
            ++s_state.m_argc;
            --argc;
            ++argv;
        }

        if (argc == 1)
            return false;
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " "
                      << "dimx dimy dimz tsteps\n\twhere args are integer sizes of the data fields and tsteps "
                         "is the number of time steps to run in a benchmark run"
                      << std::endl;
            exit(1);
        }

        for (size_t i = 0; i < 3; ++i)
            s_state.m_d[i] = std::atoi(argv[i + 1]);
        s_state.m_steps = argc > 4 ? std::atoi(argv[4]) : 10;
        s_state.m_needs_verification = argc < 6 || std::strcmp(argv[5], "-d") != 0;
        return true;
    }

    struct patterns {
        std::vector<std::string> positives;
        std::vector<std::string> negatives = {};
    };

    std::vector<std::string> split_patterns(std::string src) {
        std::vector<std::string> res;
        while (true) {
            auto column_pos = src.find(":");
            if (column_pos != 0)
                res.push_back(src.substr(0, column_pos));
            if (column_pos == std::string::npos)
                break;
            src = src.substr(column_pos + 1);
        }
        return res;
    }

    std::string merge_patterns(std::vector<std::string> const &src) {
        std::string res;
        for (auto &&item : src) {
            if (!res.empty())
                res += ":";
            res += item;
        }
        return res;
    }

    patterns parse_patterns(std::string const &flag) {
        auto dash_pos = flag.find("-");
        patterns res = {split_patterns(flag.substr(0, dash_pos))};
        if (dash_pos != std::string::npos)
            res.negatives = split_patterns(flag.substr(dash_pos + 1));
        return res;
    }

    std::string to_string(patterns const &src) {
        std::string res = merge_patterns(src.positives);
        if (!src.negatives.empty())
            res += "-" + merge_patterns(src.negatives);
        return res;
    }

    class separator {
        std::string m_val;
        bool m_is_first;

        friend std::ostream &operator<<(std::ostream &strm, separator &obj) {
            if (obj.m_is_first) {
                obj.m_is_first = false;
                return strm;
            }
            return strm << obj.m_val;
        }

      public:
        separator(std::string val) : m_val(std::move(val)), m_is_first(true) {}
    };

    class perf_times {
        using key_t = std::tuple<std::string, std::string, std::string>;
        using value_t = std::vector<double>;
        using map_t = std::map<key_t, value_t>;

        map_t m_map;

        friend std::ostream &operator<<(std::ostream &strm, perf_times const &obj) {
            strm << "{\n";
            strm << "  \"outputs\" : [";
            int outputs = 0;
            for (auto &&item : obj.m_map) {
                if (outputs)
                    strm << ",";
                strm << "\n    {\n";
                strm << "      \"name\" : \"" << std::get<0>(item.first) << "\",\n";
                strm << "      \"backend\" : \"" << std::get<1>(item.first) << "\",\n";
                strm << "      \"float_type\" : \"" << std::get<2>(item.first) << "\",\n";
                strm << "      \"series\" : [";
                int series = 0;
                for (auto val : item.second) {
                    if (series)
                        strm << ", ";
                    strm << val;
                    ++series;
                }
                strm << "]\n";
                strm << "    }";
                ++outputs;
            }
            if (outputs)
                strm << "\n  ";
            strm << "]\n";
            strm << "}\n";
            return strm;
        }

      public:
        void add(std::string const &name, std::string const &backend, std::string const &float_type, double time) {
            m_map[key_t(name, backend, float_type)].push_back(time);
        }
    };

    auto &times() {
        static perf_times res;
        return res;
    }
} // namespace

namespace gridtools {
    namespace test_environment_impl_ {
        void add_time(std::string const &name, std::string const &backend, std::string const &float_type, double time) {
            times().add(name, backend, float_type, time);
        }

        int cmdline_params::d(size_t i) { return s_state.m_d[i]; }
        size_t cmdline_params::steps() { return s_state.m_steps; }
        bool cmdline_params::needs_verification() { return s_state.m_needs_verification; }
        int &cmdline_params::argc() { return s_state.m_argc; }
        char **cmdline_params::argv() { return s_state.m_argv; }

        void flush_cache(timer_omp const &) {
            static std::size_t n = 1024 * 1024 * 21 / 2;
            static std::vector<double> a_(n), b_(n), c_(n);
            double *a = a_.data();
            double *b = b_.data();
            double *c = c_.data();
#pragma omp parallel for
            for (std::size_t i = 0; i < n; i++)
                a[i] = b[i] * c[i];
        }
    } // namespace test_environment_impl_
} // namespace gridtools

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    auto patterns = parse_patterns(testing::FLAGS_gtest_filter);
    bool perf_mode = init(argc, argv);
    if (perf_mode) {
        patterns.negatives.emplace_back("*/*_domain_size_*.*");
        if (!s_state.m_needs_verification) {
            auto &&listeners = ::testing::UnitTest::GetInstance()->listeners();
            delete listeners.Release(listeners.default_result_printer());
        }
    } else {
        patterns.negatives.emplace_back("*/*_cmdline.*");
    }
    testing::FLAGS_gtest_filter = to_string(patterns);
    int res = RUN_ALL_TESTS();
    if (perf_mode && res == 0)
        std::cout << times();
    return res;
}
