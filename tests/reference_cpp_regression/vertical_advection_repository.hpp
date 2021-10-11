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

#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include <gridtools/common/defs.hpp>

// define some physical constants
constexpr double BETA_V = 0;
constexpr double BET_M = (1 - BETA_V) / 2;
constexpr double BET_P = (1 + BETA_V) / 2;

namespace gridtools {
    class vertical_advection_repository {

        using column_t = std::vector<double>;

        template <class Fun>
        struct cached_f {
            Fun m_fun;
            mutable std::map<std::array<int_t, 2>, column_t> m_cache;

            column_t const *lookup(int_t i, int_t j) const {
                column_t const *res = nullptr;
#pragma omp critical
                {
                    auto it = m_cache.find({i, j});
                    if (it != m_cache.end())
                        res = &it->second;
                }
                return res;
            }

            double operator()(int_t i, int_t j, int_t k) const {
                if (auto *cached = lookup(i, j))
                    return (*cached)[k];
                auto column = m_fun(i, j);
                double res = column[k];
#pragma omp critical
                { m_cache.insert({{i, j}, std::move(column)}); }
                return res;
            }
        };
        template <class Fun>
        static cached_f<Fun> cache(Fun &&fun) {
            return {std::forward<Fun>(fun)};
        }

        const double PI = std::atan(1) * 4;
        size_t m_d1, m_d2, m_d3;

        double x(int_t i) const { return 1. * i / m_d1; }
        double y(int_t j) const { return 1. * j / m_d2; }
        double z(int_t k) const { return 1. * k / m_d3; }

      public:
        using fun_t = std::function<double(int_t, int_t, int_t)>;

        double dtr_stage = 3. / 20.;

        const fun_t u_stage = [this](int_t i, int_t j, int_t) {
            double t = x(i) + y(j);
            // u values between 5 and 9
            return 7 + std::cos(PI * t) + std::sin(2 * PI * t);
        };

        const fun_t wcon = [this](int_t i, int_t j, int_t k) {
            // wcon values between -2e-4 and 2e-4 (with zero at k=0)
            return 2e-4 * (-1.07 + (2 + cos(PI * (x(i) + z(k))) + cos(PI * y(j))) / 2);
        };

        const fun_t utens = [this](int_t i, int_t j, int_t k) {
            // utens values between -3e-6 and 3e-6 (with zero at k=0)
            return 3e-6 * (-1.0235 + (2. + cos(PI * (x(i) + y(j))) + cos(PI * y(j) * z(k))) / 2);
        };

        const fun_t u_pos = u_stage;

        const fun_t utens_stage_in = [this](int_t i, int_t j, int_t k) {
            double t = x(i) + y(j);
            return 7 + 1.25 * (2. + cos(PI * t) + sin(2 * PI * t)) + .1 * k;
        };

        const fun_t utens_stage_out = cache([this](int_t i, int_t j) {
            constexpr int_t IShift = 1;
            constexpr int_t JShift = 0;
            column_t c(m_d3), d(m_d3), res(m_d3);
            // forward
            // k minimum
            int k = 0;
            {
                double gcv = .25 * (wcon(i + IShift, j + JShift, k + 1) + wcon(i, j, k + 1));
                double cs = gcv * BET_M;
                c[k] = gcv * BET_P;
                double b = dtr_stage - c[k];
                // update the d column
                double correctionTerm = -cs * (u_stage(i, j, k + 1) - u_stage(i, j, k));
                d[k] = dtr_stage * u_pos(i, j, k) + utens(i, j, k) + utens_stage_in(i, j, k) + correctionTerm;
                c[k] /= b;
                d[k] /= b;
            }
            // kbody
            for (++k; k < m_d3 - 1; ++k) {
                double gav = -.25 * (wcon(i + IShift, j + JShift, k) + wcon(i, j, k));
                double gcv = .25 * (wcon(i + IShift, j + JShift, k + 1) + wcon(i, j, k + 1));
                double as = gav * BET_M;
                double cs = gcv * BET_M;
                double a = gav * BET_P;
                c[k] = gcv * BET_P;
                double bcol = dtr_stage - a - c[k];
                double correctionTerm =
                    -as * (u_stage(i, j, k - 1) - u_stage(i, j, k)) - cs * (u_stage(i, j, k + 1) - u_stage(i, j, k));
                d[k] = dtr_stage * u_pos(i, j, k) + utens(i, j, k) + utens_stage_in(i, j, k) + correctionTerm;
                double divided = 1 / (bcol - (c[k - 1] * a));
                c[k] *= divided;
                d[k] = (d[k] - d[k - 1] * a) * divided;
            }
            // k maximum
            {
                double gav = -.25 * (wcon(i + IShift, j + JShift, k) + wcon(i, j, k));
                double as = gav * BET_M;
                double a = gav * BET_P;
                double b = dtr_stage - a;
                // update the d column
                double correctionTerm = -as * (u_stage(i, j, k - 1) - u_stage(i, j, k));
                d[k] = dtr_stage * u_pos(i, j, k) + utens(i, j, k) + utens_stage_in(i, j, k) + correctionTerm;
                d[k] = (d[k] - d[k - 1] * a) / (b - (c[k - 1] * a));
            }
            // backward
            double data = d[k];
            c[k] = data;
            res[k] = dtr_stage * (data - u_pos(i, j, k));
            // kbody
            for (--k; k >= 0; --k) {
                data = d[k] - c[k] * data;
                c[k] = data;
                res[k] = dtr_stage * (data - u_pos(i, j, k));
            }
            return res;
        });

        vertical_advection_repository(size_t d1, size_t d2, size_t d3) : m_d1(d1), m_d2(d2), m_d3(d3) {}
    };
} // namespace gridtools
