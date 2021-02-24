#include "driver.hpp"
#include "nabla_cuda.hpp"
#include <gtest/gtest.h>

TEST(fvm, nabla_cuda) { fvm_nabla_driver(nabla); }
