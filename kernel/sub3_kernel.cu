#include <stdio.h>
#include <assert.h>
#include <algorithm>

// only responsible for aggr along a single direction
__global__ void cost_volume_kernel(float *c,
                            const float *a,
                            const float *b,
                            // float *disparity_map,
                            // float *L,
                            // float *L_min,
                            // float *L_min_disp,
                            const int h,
                            const int w,
                            const int d_range,
                            const int d_min
                            ) {

    int row = blockIdx.x; // h 
    int col = blockIdx.y; // w
    int disp = blockIdx.z * blockDim.z + threadIdx.z; // d_range: d_min to d_max
    // printf("row: %d | col: %d | disp: %d | \n", row, col, disp);
   
    int a_i = row * w + col;
    int b_i = row * w + (col - (d_min+disp));// ** IMPORTANT: now is not calculating every pixel
    
    int c_i = a_i * d_range + disp; // row * w * d_range + col * d_range + disp;
    if (col-(d_min+disp)>=0){
        c[c_i] = abs(a[a_i] - b[b_i]); // init cost volume before aggregation
    }else{
        c[c_i] = 99.9; // set to infinity float num if the pixel does not exist in b
    }
    
    
    __syncthreads();
}

__global__ void cost_aggr_L2R_kernel(
                            const float *c,
                            // const float *a,
                            // const float *b,
                            // float *disparity_map,
                            float *L,
                            float *L_min,
                            float *L_min_disp,
                            const int h,
                            const int w,
                            const int d_range,
                            const int d_min,
                            const float p1,
                            const float p2,
                            const int r

                            ) {

    int row = blockIdx.x; // h 
    int col = blockIdx.y; // w
    int disp = blockIdx.z * blockDim.z + threadIdx.z; // d_range: d_min to d_max
    // printf("row: %d | col: %d | disp: %d | \n", row, col, disp);
    
    // FIXME: hard-coded h*w now, change it to passed in parameters
    int a_i = row * w + col;
    // int b_i = row * w + (col - (d_min+disp));// ** IMPORTANT: now is not calculating every pixel
    // // TODO: what if b_i <0?? 
    int c_i = a_i * d_range + disp; // row * w * d_range + col * d_range + disp;
   
    
    
   // regularization term, hyper parameter
    // aggr from l to r.
    // since it's accumulative, cannot do parallel along dispth dim (one direciton always using "-r", no "+r". The agg is "+/- disp")
    // int r; // L->R. If R->L, r=-1; if Up->Down, r=w.
    if (disp == 0){ // avoid competition, for each point a_i
        L[c_i] = c[c_i]; // disp = 0 has no less disp to calculate L // different from the col=0 init 
        L_min[a_i] = c[c_i]; // init the min value of pixel a_i
        // printf("a_i %f\n", L_min[a_i]);
        L_min_disp[a_i] = d_min;
        // disparity_map[a_i] = d_min;
        // printf("%f\n",L[0]);
        
    }
    
    __syncthreads(); // sync before cost aggregation
    if (col==0){ // AVOID competition, col depends on (col-r)
    //     // init all disp cost for col=0, then do calculation from col=1 (below fors-loop)
    //     // r = 1;
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
                    L_min_disp[cur_pixel] = d_l+d_min;
                }
            }
        
        }

    }
    
    __syncthreads();
}

__global__ void disparity_calculation_kernel(
                            float *disparity_map,
                            float *L,
                            float *L_min,
                            float *L_min_disp,
                            const int h,
                            const int w,
                            const int d_range,
                            const int d_min){
    int row = blockIdx.x; // h 
    int col = blockIdx.y; // w
    int disp = blockIdx.z * blockDim.z + threadIdx.z;

    int a_i = row * w + col;

    if (disp == 0){ // avoid competition (equivalent to threadIdx.z==0)

        disparity_map[a_i] = L_min_disp[a_i];
    }
    
    
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
    printf("here CPU of lauch_sub3\n");

    // calculate raw cost vol
    

    // declare all the __shared__ here:
    // mem alloc can be parallelled with cost vol calculation
    float* L = new float[h*w*d_range];
    float *d_L;
    auto isSuccess = cudaMalloc((void**)&d_L, sizeof(float)*h*w*d_range);
    // printf("Is success %d\n", isSuccess);
   
    float* L_min = new float[h*w];
    float *d_L_min;
    isSuccess = cudaMalloc((void**)&d_L_min, sizeof(float)*h*w);
    // printf("Is success %d\n", isSuccess); 
  
    float* L_min_disp = new float[h*w];
    float *d_L_min_disp;
    isSuccess = cudaMalloc((void**)&d_L_min_disp, sizeof(float)*h*w);
    // printf("Is success %d\n", isSuccess); 

    // 1st kernel
    cost_volume_kernel<<<grid, block>>>(c, a, b, h, w, d_range, d_min);
    cudaError_t err_1 = cudaGetLastError();        // Get error code
    if ( err_1 != cudaSuccess )
    {
        printf("cost_volume success CUDA Error: %s\n", cudaGetErrorString(err_1));
        exit(-1);
    }
    else{
        printf("cost_volume success!\n");
    }
    cudaDeviceSynchronize();
    
    
    // hyper params
    float p1=10, p2=150; 
    int r = 1;
    cudaDeviceSynchronize();
    dim3 grid_2(h,w,1);
    dim3 block_2(1,1,d_range);
    // 2nd kernel
    cost_aggr_L2R_kernel<<<grid_2, block_2>>>(c, d_L, d_L_min, d_L_min_disp, h, w, d_range, d_min, p1, p2, r);
    cudaDeviceSynchronize();
    cudaError_t err_2 = cudaGetLastError();        // Get error code
    if ( err_2 != cudaSuccess )
    {
        printf("cost_aggr_L2R_kernel CUDA Error: %s\n", cudaGetErrorString(err_2));
        exit(-1);
    }
    else{
        printf("cost_aggr_L2R_kernel success!\n");
    }
    
    // 3rd kernel
    disparity_calculation_kernel<<<grid, block>>>(d_disparity, d_L, d_L_min, d_L_min_disp, h, w, d_range, d_min);
    cudaError_t err_3 = cudaGetLastError();        // Get error code
    if ( err_3 != cudaSuccess )
    {
        printf("disparity_calculation_kernel CUDA Error: %s\n", cudaGetErrorString(err_3));
        exit(-1);
    }
    else{
        printf("disparity_calculation_kernel success!\n");
    }
    cudaDeviceSynchronize();

    // free all the shared variables
    cudaFree(d_L);
    cudaFree(d_L_min);
    cudaFree(d_L_min_disp);

    /* below cpy is discarded, since the cpy is already handled by some pytorch lib*/
    // cudaMemcpy(disparity_map, d_disparity, sizeof(int)*H*W, cudaMemcpyDeviceToHost);
    
}
