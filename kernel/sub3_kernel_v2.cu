/*
improvement from v2:    
    remove all the __shared__, pass them as parameters from Host code, 
    since kernel code cannot declare array with variable size, but host can.
    Therefore can pass in H W D_range to host.
*/
#include <stdio.h>
#include <algorithm>
// add 3D matrix subtraction 
__global__ void sub3_kernel(float *c,
                            const float *a,
                            const float *b,
                            float *disparity_map
                            // const int h,
                            // const int w,
                            ) {
    // printf("here GPU \n");
    const int h = 2;
    const int w = 4;
    const int d_range = 4;
    // __shared__ float* shared_min_value = new float[h*w];
    __shared__ float shared_min_value[h * w]; // stored the min among dispth dim
    __shared__ int shared_min_disp[h * w]; // disprity map

    int row = blockIdx.x * blockDim.x + threadIdx.x; // h 
    int col = blockIdx.y * blockDim.y + threadIdx.y; // w
    int disp = blockIdx.z * blockDim.z + threadIdx.z; // d_range
    // printf("row: %d | col: %d | disp: %d | \n", row, col, disp);
    
    // FIXME: hard-coded h*w now, change it to passed in parameters
    int a_i = row * w + col;
    int b_i = row * w + (col - disp);// ** IMPORTANT: now is not calculating every pixel
    // TODO: what if b_i <0?? 
    int c_i = a_i * d_range + disp; // row * w * d_range + col * d_range + disp;
    if (col-disp>=0){
        c[c_i] = abs(a[a_i] - b[b_i]); // init cost volume before aggregation
    }else{
        c[c_i] = 99.9; // set to infinity float num if the pixel does not exist in b
    }
    
    // __syncthreads(); // sync before cost aggregation
    __shared__ float L[h * w * d_range];
    __shared__ float L_min[h * w];
    __shared__ float L_min_disp[h * w]; // save the corresponding disp to get min_L
    
    float p1=0, p2=0; // regularization term, hyper parameter
    // aggr from l to r.
    // since it's accumulative, cannot do parallel along dispth dim (one direciton always using "-r", no "+r". The agg is "+/- disp")
    int r=1; // L->R. If R->L, r=-1; if Up->Down, r=w.
    if (disp == 0){ // avoid competition, for each point a_i
        L[c_i] = c[c_i]; // disp = 0 has no less disp to calculate L // different from the col=0 init 
        L_min[a_i] = c[c_i]; // init the min value of pixel a_i
        // printf("a_i %f\n", L_min[a_i]);
        L_min_disp[a_i] = disp;
    }
    
    __syncthreads(); // sync before cost aggregation
    if (col==0){ // AVOID competition, col depends on (col-r)
        // init all disp cost for col=0, then do calculation from col=1 (below for-loop)
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
                    L[last_pixel*d_range+d_l],
                    fminf(
                        L[last_pixel*d_range+d_l-1]+p1,
                        fminf(L[last_pixel*d_range+d_l+1]+p1, L_min[last_pixel]+p2 )
                    )
                ) 
                - (L_min[last_pixel]); // minus this term to bound aggr cost, constant for all disp

                if (L[cur_pixel*d_range+d_l] < L_min[cur_pixel]){ // FIXME: "<" or "≤"?
                    L_min[cur_pixel] = L[cur_pixel*d_range+d_l];
                    L_min_disp[cur_pixel] = d_l;
                }
            }
            
        
        }

    }
    
    __syncthreads(); 
    // copy back the shared disparity map to return to cpu
    if (disp == 0){ // avoid competition (equivalent to threadIdx.z==0)
        // disparity[a_i]=shared_min_disp[a_i];
        disparity_map[a_i]=L_min_disp[a_i];

    }

}

void launch_sub3(float* c,
                 const float* a,
                 const float* b,
                 float* d_disparity
                //  const int H,
                //  const int W
                 ) {

    dim3 grid(1,1,1);
    dim3 block(2,4,4);

    // const int H = 2;
    // const int W = 4;
    // int H=2, W=4;
    // float* try_array = new float[H*W];
    // float *d_try;
    // cudaMalloc((void**)&d_try, sizeof(float)*H*W);

    // int disparity_map[H*W];
    // int *d_disparity;
    // cudaMalloc((void**)&d_disparity, sizeof(int)*H*W);

    sub3_kernel<<<grid, block>>>(c, a, b, d_disparity);
    cudaDeviceSynchronize();
    // cudaMemcpy(disparity_map, d_disparity, sizeof(int)*H*W, cudaMemcpyDeviceToHost);
    printf("here CPU of lauch_sub3\n");

    /* below accesssing c will cause core dump*/
    // for (int row=0; row<H; row++){
    //     for (int col=0; col<W; col++){
    //         for (int disp=0; disp<W; disp++){
    //             printf("%f \t", c[row*W*W + col*W + disp]);
    //         }
    //         printf("\n");
    //         // printf("%f \t", b[row*W+ col]);
    //     }
    //     printf("-----------\n");
    // }
    /*-----------core dump ends------------*/
    // printf("\n----disparity map-------\n");
    // for (int row=0; row<H; row++){
    //     for (int col=0; col<W; col++){
    //         printf("%d \t", disparity_map[row*W+ col]);
    //     }
    //     printf("\n");
    // }
    // printf("------end-----\n");

}

/* below is pure nvcc compile style, not with pytorch*/
/*
int main(void){
    const int H = 2;
    const int W = 4;
  
    const float a[H*W] = {1,2,3,4,
                        5,6,7,8};
    const float b[H*W] = {2,3,4,5,
                        6,7,8,9};
    float c[H*W*W];
    int disparity[H*W];


    float *d_a, *d_b, *d_c;
    int *d_disparity;

    cudaMalloc((void**)&d_a, sizeof(float)*H*W);
    cudaMalloc((void**)&d_b, sizeof(float)*H*W);
    cudaMalloc((void**)&d_c, sizeof(float)*H*W*W);
    cudaMalloc((void**)&d_disparity, sizeof(int)*H*W);
    
    cudaMemcpy(d_a, a, sizeof(float)*H*W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*H*W, cudaMemcpyHostToDevice);
    
    printf("here CPU \n");

    dim3 grid(1,1,1);
    dim3 block(2,4,4);
    sub3_kernel<<<grid, block>>>(d_c, d_a, d_b, d_disparity);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, sizeof(float)*H*W*W, cudaMemcpyDeviceToHost);
    cudaMemcpy(disparity, d_disparity, sizeof(int)*H*W, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_disparity);
    for (int row=0; row<H; row++){
        for (int col=0; col<W; col++){
            for (int disp=0; disp<W; disp++){
                printf("%f \t", c[row*W*W + col*W + disp]);
            }
            printf("\n");
            // printf("%f \t", b[row*W+ col]);
        }
        printf("-----------\n");
    }
    printf("\n----disparity map-------\n");
    for (int row=0; row<H; row++){
        for (int col=0; col<W; col++){
            printf("%d \t", disparity[row*W+ col]);
        }
        printf("\n");
    }
    printf("------end-----\n");
*/