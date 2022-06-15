// __global__ void add2_kernel(float* c,
//                             const float* a,
//                             const float* b,
//                             int n) {
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
//             i < n; i += gridDim.x * blockDim.x) {
//         c[i] = a[i] + b[i];
//     }
// }

// void launch_add2(float* c,
//                  const float* a,
//                  const float* b,
//                  int n) {
//     dim3 grid((n + 1023) / 1024);
//     dim3 block(1024);
//     add2_kernel<<<grid, block>>>(c, a, b, n);
// }
#include "stdio.h"
// add 3D matrix subtraction 
__global__ void sub3_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n
                            ) {
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    //         i < n; i += gridDim.x * blockDim.x) {
    //     c[i] = a[i] + b[i];
    // }
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;
    // std::cout << "row:"<< row << " col:" << col << "dep:" << dep;
    printf("row: %d | col: %d | dep: %d | \n", row, col, dep);
    
    // FIXME: hard-coded h*w now, change it to passed in parameters
    int h = 2;
    int w = 4;
    int c_i = row * w * w + col * w + dep;
    int a_i = row * w + col;
    int b_i = row * w + dep;
    c[c_i] = a[a_i] - b[b_i];
}

void launch_sub3(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    //  FIXME: modify the dim to adapt to 3D tensors
    dim3 grid(1,1,1);
    dim3 block(2,4,4);
    sub3_kernel<<<grid, block>>>(c, a, b, n);
}
