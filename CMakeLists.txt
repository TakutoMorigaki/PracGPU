cmake_minimum_required(VERSION 3.15)
project(game2048_env)

set(pybind11_DIR "C:/Users/taku1/Programing/PracGPU/torch_gpu/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

pybind11_add_module(
    game2048_env

    bindings.cpp
    board.cpp
    env2048.cpp
)