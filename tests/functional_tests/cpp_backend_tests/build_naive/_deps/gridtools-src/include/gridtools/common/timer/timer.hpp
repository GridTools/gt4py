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
#include <sstream>
#include <string>
#include <utility>

namespace gridtools {

    /**
     * @class Timer
     * Measures total elapsed time between all start and stop calls
     */
    template <class Impl>
    class timer {
      private:
        std::string m_name;
        double m_total_time = 0;
        size_t m_counter = 0;
        Impl m_impl;

      public:
        timer() = default;
        timer(std::string name) : m_name(std::move(name)) {}

        /**
         * Reset counters
         */
        void reset() {
            m_total_time = 0;
            m_counter = 0;
        }

        /**
         * Start the stop watch
         */
        void start() { m_impl.start_impl(); }

        /**
         * Pause the stop watch
         */
        void pause() {
            m_total_time += m_impl.pause_impl();
            m_counter++;
        }

        /**
         * @return total elapsed time [s]
         */
        double total_time() const { return m_total_time; }

        /**
         * @return how often the timer was paused
         */
        size_t count() const { return m_counter; }

        /**
         * @return total elapsed time [s] as string
         */
        std::string to_string() const {
            std::ostringstream out;
            if (m_total_time < 0 || std::isnan(m_total_time))
                out << m_name << "\t[s]\t"
                    << "NO_TIMES_AVAILABLE"
                    << " (" << m_counter << "x called)";
            else
                out << m_name << "\t[s]\t" << m_total_time << " (" << m_counter << "x called)";
            return out.str();
        }
    };
} // namespace gridtools
