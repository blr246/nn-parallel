cmake_minimum_required(VERSION 2.8)

# Adding GTest to build as per recommendation of GTest devs. See
# https://groups.google.com/forum/#!msg/googletestframework/Zo7_HOv1MJ0/F4ZBGjh_ePcJ
# for more information.
if(WITH_GTEST)
  set(gtest_force_shared_crt "ON" CACHE BOOL "Use shared crt." FORCE)
  add_subdirectory(gtest-1.6.0)
  set(GTEST_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/gtest-1.6.0/include PARENT_SCOPE)
  enable_testing()
endif(WITH_GTEST)
