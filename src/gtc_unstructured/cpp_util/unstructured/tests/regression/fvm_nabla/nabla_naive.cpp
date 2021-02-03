#include "nabla_naive.hpp"
#include "driver.hpp"
#include <gtest/gtest.h>

TEST(fvm, nabla_naive) { fvm_nabla_driver(nabla); }
