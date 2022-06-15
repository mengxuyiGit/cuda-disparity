#include <torch/extension.h>
#include "sub3.h"

// void torch_launch_sub3(torch::Tensor &c,
//                        const torch::Tensor &a,
//                        const torch::Tensor &b,
//                        int64_t n) {
//     launch_sub3((float *)c.data_ptr(),
//                 (const float *)a.data_ptr(),
//                 (const float *)b.data_ptr(),
//                 n);
// }
void torch_launch_sub3(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       torch::Tensor &d,
                       int64_t h,
                       int64_t w,
                       int64_t d_range,
                       int64_t d_min) {
    launch_sub3((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                (float *)d.data_ptr(),
                (int) h,
                (int) w,
                (int) d_range,
                (int) d_min);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_sub3",
          &torch_launch_sub3,
          "sub3 kernel warpper");
}

TORCH_LIBRARY(sub3, m) {
    m.def("torch_launch_sub3", torch_launch_sub3);
}