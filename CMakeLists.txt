cmake_minimum_required(VERSION 2.8)
project(blr_nn_all)

option(WITH_GTEST "Enable GTests." ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message("Setting CMAKE_INSTALL_PREFIX to ${CMAKE_BINARY_DIR}.")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
else(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message("CMAKE_INSTALL_PREFIX was set to ${CMAKE_INSTALL_PREFIX}.")
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Enable OpenMP in UNIX.
# Enable multicore builds in MSVC.
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
if(WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI /W4")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W4")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif(WIN32)

# Make sure that OpenCV_DIR is set in environment or find_package() will fail.
find_package(OpenCV REQUIRED)

add_subdirectory(thirdparty)
add_subdirectory(src)
