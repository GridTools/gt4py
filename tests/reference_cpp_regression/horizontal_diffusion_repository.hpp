/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cmath>
#include <functional>

#include <gridtools/common/defs.hpp>

namespace gridtools {
    class horizontal_diffusion_repository {
        using fun_t = std::function<double(int_t, int_t, int_t)>;
        using j_fun_t = std::function<double(int_t)>;
        uint_t m_d1, m_d2, m_d3;

        double m_delta0 = (0.995156 - 0.994954) / (m_d2 - 1.);
        double m_delta1 = (0.995143 - 0.994924) / (m_d2 - 1.);

        j_fun_t m_crlat0 = [this](int_t j) { return 0.994954 + j * m_delta0; };
        j_fun_t m_crlat1 = [this](int_t j) { return 0.994924 + j * m_delta1; };

      public:
        horizontal_diffusion_repository(uint_t d1, uint_t d2, uint_t d3) : m_d1(d1), m_d2(d2), m_d3(d3) {}

        fun_t in = [this](int_t i, int_t j, int_t) {
            const double PI = std::atan(1.) * 4.;
            double dx = 1. / m_d1;
            double dy = 1. / m_d2;
            double x = dx * i;
            double y = dy * j;
            // in values between 5 and 9
            return 5. + 8 * (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;
        };

        fun_t coeff = [](int_t, int_t, int_t) { return 0.025; };

        fun_t crlato = [this](int_t, int_t j, int_t) { return m_crlat1(j) / m_crlat0(j); };

        fun_t crlatu = [this](int_t, int_t j, int_t) { return j == 0 ? 0 : m_crlat1(j - 1) / m_crlat0(j); };

        fun_t out = [this](int_t i, int_t j, int_t k) {
            auto lap = [=](int_t ii, int_t jj) {
                return 4 * in(ii, jj, k) -
                       (in(ii + 1, jj, k) + in(ii, jj + 1, k) + in(ii - 1, jj, k) + in(ii, jj - 1, k));
            };
            auto flx = [=](int_t ii, int_t jj) {
                double res = lap(ii + 1, jj) - lap(ii, jj);
                if (res * (in(ii + 1, jj, k) - in(ii, jj, k)) > 0)
                    res = 0.;
                return res;
            };
            auto fly = [=](int_t ii, int_t jj) {
                auto res = lap(ii, jj + 1) - lap(ii, jj);
                if (res * (in(ii, jj + 1, k) - in(ii, jj, k)) > 0)
                    res = 0.;
                return res;
            };
            return in(i, j, k) - coeff(i, j, k) * (flx(i, j) - flx(i - 1, j) + fly(i, j) - fly(i, j - 1));
        };

        fun_t out_simple = [this](int_t i, int_t j, int_t k) {
            auto lap = [=](int_t ii, int_t jj) {
                return in(ii + 1, jj, k) + in(ii - 1, jj, k) - 2 * in(ii, jj, k) +
                       crlato(ii, jj, k) * (in(ii, jj + 1, k) - in(ii, jj, k)) +
                       crlatu(ii, jj, k) * (in(ii, jj - 1, k) - in(ii, jj, k));
            };
            auto fluxx = lap(i + 1, j) - lap(i, j);
            auto fluxx_m = lap(i, j) - lap(i - 1, j);
            auto fluxy = crlato(i, j, k) * (lap(i, j + 1) - lap(i, j));
            auto fluxy_m = crlato(i, j, k) * (lap(i, j) - lap(i, j - 1));
            return in(i, j, k) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * coeff(i, j, k);
        };
    };
} // namespace gridtools
