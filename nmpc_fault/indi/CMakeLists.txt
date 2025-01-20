cmake_minimum_required(VERSION 3.4...3.18)
project(
  indi
  VERSION 1.0
  LANGUAGES CXX)

add_subdirectory(pybind11)
pybind11_add_module(indi indi.cpp)

include_directories(include)
target_compile_definitions(indi
							PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
