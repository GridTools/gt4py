#include "${CMAKE_CURRENT_SOURCE_DIR}/fvm_nabla_driver.hpp"
#include "${STENCIL_IMPL_SOURCE}"
#include <gtest/gtest.h>

TEST(fvm, nabla_${BACKEND}) { fvm_nabla_driver(nabla); }
