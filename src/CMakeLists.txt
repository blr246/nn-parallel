cmake_minimum_required(VERSION 2.8)

file(GLOB HEADERS "*.h" "*.hpp")

# Library targets:
#   nn_core
project(nn_parallel_core)
set(SRCS
    idx_cv.cpp
    ${HEADERS}
    )
add_library(nn_parallel_core STATIC ${SRCS} ${HEADERS})

if(WIN32)
  # Set max warning level, disable c99 initialization warning,
  # disable unsafe use of bool for STATIC_ASSERT.
  set_target_properties(nn_parallel_core PROPERTIES COMPILE_FLAGS "/W4")
  add_definitions("/wd4351;4804")
endif(WIN32)

project(nn_parallel)
set(SRCS
    idx_cv.cpp
    neural_network.cpp
    ${HEADERS}
    )

add_executable(nn_parallel ${SRCS})
target_link_libraries(nn_parallel ${OpenCV_LIBS})

if(WIN32)
  # Set max warning level, disable c99 initialization warning,
  # disable unsafe use of bool for STATIC_ASSERT.
  set_target_properties(nn_parallel PROPERTIES COMPILE_FLAGS "/W4")
  add_definitions("/wd4351;4804")
endif(WIN32)

# Copy data files to build directory.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(WITH_GTEST)
  project(nn_parallel_gtest)
  set(SRCS
      nn_parallel_gtest.cpp
      ${HEADERS}
      )
  include_directories(${GTEST_INCLUDE_DIRS})
  add_executable(nn_parallel_gtest ${SRCS} ${HEADERS})
  target_link_libraries(nn_parallel_gtest nn_parallel_core ${OpenCV_LIBS} gtest)
  add_test(nn_parallel_gtest ${CMAKE_CURRENT_BINARY_DIR}/nn_parallel_gtest)
  if(WIN32)
    # Set max warning level, disable c99 initialization warning,
    # disable unsafe use of bool for STATIC_ASSERT.
    set_target_properties(nn_parallel_gtest PROPERTIES COMPILE_FLAGS "/W4")
    add_definitions("/wd4351;4804")
endif(WIN32)
endif(WITH_GTEST)

project(nn_parallel_all)
