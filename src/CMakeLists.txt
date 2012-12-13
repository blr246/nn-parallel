cmake_minimum_required(VERSION 2.8)
project(dropout_nn_parallel)

file(GLOB HEADERS "*.h" "*.hpp")

set(SRCS
    neural_network.cpp
    ${HEADERS}
    )

add_executable(dropout_nn_parallel ${SRCS})
target_link_libraries(dropout_nn_parallel ${OpenCV_LIBS})

if(WIN32)
  # Set max warning level, disable c99 initialization warning,
  # disable unsafe use of bool for STATIC_ASSERT.
  set_target_properties(dropout_nn_parallel PROPERTIES COMPILE_FLAGS "/W4")
  add_definitions("/wd4351;4804")
endif(WIN32)

if(WITH_GTEST)
  project(dropout_nn_parallel_gtest)
  set(SRCS
      "dropout_nn_parallel_gtest.cpp")
  include_directories(${GTEST_INCLUDE_DIRS})
  add_executable(dropout_nn_parallel_gtest ${SRCS} ${HEADERS})
  target_link_libraries(dropout_nn_parallel_gtest ${OpenCV_LIBS} gtest)
  add_test(dropout_nn_parallel_gtest dropout_nn_parallel_gtest)
endif(WITH_GTEST)

project(dropout_nn_parallel_all)
