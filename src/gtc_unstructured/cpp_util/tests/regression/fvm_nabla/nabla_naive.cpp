#include "driver.hpp"
#include "nabla_unaive.hpp"
#include <gtest/gtest.h>

TEST(fvm, nabla_naive) { fvm_nabla_driver(nabla); }
