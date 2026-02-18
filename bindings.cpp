#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "env2048.hpp"

namespace py = pybind11;

PYBIND11_MODULE(game2048_env, m){
    py::class_<Env2048>(m, "Env2048")

        .def(py::init<>())

        .def("reset", &Env2048::reset)

        .def("step",
            [](Env2048 &env, int action){
                float reward;
                bool done;

                auto state = env.Step(action, reward, done);

                return py::make_tuple(state, reward, done);
            }
        )

        .def("get_state", &Env2048::GetState);
}