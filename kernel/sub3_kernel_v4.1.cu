/* v4.1 added d_min to test.
from v4.1 onwads, there will not be a sub3_kernel: the cost calculation and aggr will be seperated into kernels */
#include <stdio.h>
#include <algorithm>
// add 3D matrix subtraction 
__global__ void sub3_kernel_try(float *c,
                            const float *a,
                            const float *b,
                            float *disparity_map,
                            float *L_min_disp){
    int row = blockIdx.x; // h 
    int col = blockIdx.y; // w
    int disp = blockIdx.z * blockDim.z + threadIdx.z;

    int a_i = row * 1024 + col;

    // printf("here try! \n");
    // if ( row == 0){ // avoid competition (equivalent to threadIdx.z==0)
    //     // disparity[a_i]=shared_min_disp[a_i];
    //     printf("a_i %f\n", a[a_i]);
    // }
    
    if (disp == 0){ // avoid competition (equivalent to threadIdx.z==0)
        // disparity[a_i]=shared_min_disp[a_i];
        
        disparity_map[a_i] = L_min_disp[a_i];

    }
    
    
}

// only responsible for aggr along a single direction
__global__ void sub3_kernel(float *c,
                            const float *a,
                            const float *b,
                            float *disparity_map,
                            float *L,
                            float *L_min,
                            // float *L_min_disp,
                            const int h,
                            const int w,
                            const int d_range,
                            const int d_min
                            ) {

    
    // int row = blockIdx.x * blockDim.x + threadIdx.x; // h 
    // int col = blockIdx.y * blockDim.y + threadIdx.y; // w
    int row = blockIdx.x; // h 
    int col = blockIdx.y; // w
    int disp = blockIdx.z * blockDim.z + threadIdx.z; // d_range: d_min to d_max
    // printf("row: %d | col: %d | disp: %d | \n", row, col, disp);
    
    // FIXME: hard-coded h*w now, change it to passed in parameters
    int a_i = row * w + col;
    int b_i = row * w + (col - (d_min+disp));// ** IMPORTANT: now is not calculating every pixel
    // TODO: what if b_i <0?? 
    int c_i = a_i * d_range + disp; // row * w * d_range + col * d_range + disp;
    if (col-(d_min+disp)>=0){
        c[c_i] = abs(a[a_i] - b[b_i]); // init cost volume before aggregation
    }else{
        c[c_i] = 99.9; // set to infinity float num if the pixel does not exist in b
    }
    
    
    float p1=10, p2=150; // regularization term, hyper parameter
    // aggr from l to r.
    // since it's accumulative, cannot do parallel along dispth dim (one direciton always using "-r", no "+r". The agg is "+/- disp")
    int r; // L->R. If R->L, r=-1; if Up->Down, r=w.
    if (disp == 0){ // avoid competition, for each point a_i
        L[c_i] = c[c_i]; // disp = 0 has no less disp to calculate L // different from the col=0 init 
        L_min[a_i] = c[c_i]; // init the min value of pixel a_i
        // printf("a_i %f\n", L_min[a_i]);
        // L_min_disp[a_i] = d_min;
        disparity_map[a_i] = d_min;
        
    }
    
    __syncthreads(); // sync before cost aggregation
    if (col==0){ // AVOID competition, col depends on (col-r)
        // init all disp cost for col=0, then do calculation from col=1 (below for-loop)
        r = 1;
        for ( int d_col=0; d_col<d_range; d_col++){
            L[(row*w+col)*d_range+d_col]=c[(row*w+col)*d_range+d_col];
        }
        
        // aggr from col=1
        int cur_pixel, last_pixel;
        for ( int w_l=1; w_l<w; w_l++){ // w_l is the col of a
            cur_pixel = row*w + w_l;
            last_pixel = cur_pixel-r;
            for( int d_l=1; d_l<d_range; d_l++){
                L[cur_pixel*d_range+d_l] = c[cur_pixel*d_range+d_l]
                + 
                fminf(
                    fminf(L[last_pixel*d_range+d_l], L[last_pixel*d_range+d_l-1]+p1),
                    fminf(L[last_pixel*d_range+d_l+1]+p1, L_min[last_pixel]+p2 )
                ) 
                - (L_min[last_pixel]); // minus this term to bound aggr cost, constant for all disp

                if (L[cur_pixel*d_range+d_l] < L_min[cur_pixel]){ // FIXME: "<" or "≤"?
                    L_min[cur_pixel] = L[cur_pixel*d_range+d_l];

                    // L_min_disp[cur_pixel] = d_l+d_min;
                    
                    disparity_map[cur_pixel] = d_l+d_min; // now the final disp_map depend on all directions
                    // L_min_disp[cur_pixel] = d_l+d_min;
                }
            }
            // printf("update disp %f map from l min\n", L_min_disp[cur_pixel]);
        
        }

    }
    
    __syncthreads();
    // copy back the shared disparity map to return to cpu
    // if (disp == 0){ // avoid competition (equivalent to threadIdx.z==0)
    //     // disparity[a_i]=shared_min_disp[a_i];
        
    //     // disparity_map[a_i] = L_min_disp[a_i];

    // }
    
}

void launch_sub3(float* c,
                 const float* a,
                 const float* b,
                 float* d_disparity,
                 int h,
                 int w,
                 int d_range,
                 int d_min
                 ) {

    dim3 grid(h,w,1);
    dim3 block(1,1,d_range); // one block cannot hold more than 1024 threads!

   
    // declare all the __shared__ here 
    float* L = new float[h*w*d_range];
    float *d_L;
    cudaMalloc((void**)&d_L, sizeof(float)*h*w*d_range);
    float* L_min = new float[h*w];
    float *d_L_min;
    cudaMalloc((void**)&d_L_min, sizeof(float)*h*w);
    // float* L_min_disp = new float[h*w];
    // float *d_L_min_disp;
    // cudaMalloc((void**)&d_L_min_disp, sizeof(float)*h*w);
    
    sub3_kernel<<<grid, block>>>(c, a, b, d_disparity, d_L, d_L_min, h, w, d_range, d_min);
    // sub3_kernel<<<grid, block>>>(c, a, b, d_disparity, d_L, d_L_min, d_L_min_disp, h, w, d_range, d_min);
    cudaDeviceSynchronize();
    // sub3_kernel_try<<<grid, block>>>(c, a, b, d_disparity, d_L_min_disp);
    // cudaDeviceSynchronize();

    // free all the shared variables
    cudaFree(d_L);
    cudaFree(d_L_min);
    // cudaFree(d_L_min_disp);

    /* below cpy is discarded, since the cpy is already handled by some pytorch lib*/
    // cudaMemcpy(disparity_map, d_disparity, sizeof(int)*H*W, cudaMemcpyDeviceToHost);
    
    printf("here CPU of lauch_sub3\n");

}
