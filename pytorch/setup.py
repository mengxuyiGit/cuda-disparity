from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name="add2",
#     include_dirs=["include"],
#     ext_modules=[
#         CUDAExtension(
#             "add2",
#             ["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"],
#         )
#     ],
#     cmdclass={
#         "build_ext": BuildExtension
#     }
# )

setup(
    name="sub3",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "sub3",
            ["pytorch/sub3_ops.cpp", "kernel/sub3_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)