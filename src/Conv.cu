#include "Conv.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#ifndef APR_USE_CUDA
#define APR_USE_CUDA
#endif
#include "misc/CudaTools.hpp"

namespace {
    void foo() {
        MeshData<float> f(0, 0, 0);
        MeshData<uint16_t > u16(0, 0, 0);
        computeConv(f, f, f);
        compute3rdPartyConv(f, f, f);
        compute3rdPartyConv(u16, u16, u16);
    }

    template<typename T>
    __global__ void conv(T *in, T *out, int xLen, int yLen, int zLen, T *kernel, int kernelWidth) {
        // Calculate yi/zi coordinates with a ghost layer
        int yi = (blockIdx.y * blockDim.y + threadIdx.y) - blockIdx.y * 2 - 1;
        int zi = (blockIdx.z * blockDim.z + threadIdx.z) - blockIdx.z * 2 - 1;
        int yio = yi;
        int zio = zi;
        if (yi > yLen || zi > zLen) return;

        const unsigned int active = __activemask();
        const int workerIdx = threadIdx.y;

        // Boundary handling (repeat boundary element)
        bool boundary = (yi < 0 || yi == yLen || zi < 0 || zi == zLen || threadIdx.y == 0 || threadIdx.y == 31 || threadIdx.z == 0 || threadIdx.z == 31);
        if (yi < 0) yi = 0;
        if (yi == yLen) yi = yLen - 1;
        if (zi < 0) zi = 0;
        if (zi == zLen) zi = zLen - 1;

        size_t offset = zi * xLen * yLen + yi;

        for (int x = -1; x <= xLen; ++x) { // with boundaries in x-dir
            //TODO: skip reading if already read data
            T v = in[offset];

            T prevElement = __shfl_sync(active, v, workerIdx + blockDim.y - 1, blockDim.y);
            T nextElement = __shfl_sync(active, v, workerIdx + 1, blockDim.y);

            printf("(%d, %d)(%d, %d, %d)[%d] = %f (%f, %f)\n", yio, zio, threadIdx.x, yi, zi, boundary, v, nextElement, prevElement);

            if (x >= 0 && x < xLen - 1) offset += yLen;
            printf("\n");
        }
    }
}

template <typename T>
void computeConv(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &kernel) {
    std::cout << "HELLO" << std::endl;

    T *dInput;
    size_t dataSize = in.mesh.size() * sizeof(T);
    cudaMalloc(&dInput, dataSize);
    cudaMemcpy(dInput, in.mesh.get(), dataSize, cudaMemcpyHostToDevice);
    T *dOutput;
    cudaMalloc(&dOutput, dataSize);
    T *dKernel;
    size_t kernelSize = kernel.mesh.size() * sizeof(T);
    cudaMalloc(&dKernel, kernelSize);
    cudaMemcpy(dKernel, kernel.mesh.get(), kernelSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1, 32, 32);
    dim3 numBlocks(1,
                   (in.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   (in.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    conv <<<numBlocks, threadsPerBlock>>> (dInput, dOutput, in.x_num, in.y_num, in.z_num, dKernel, kernel.x_num);
    waitForCuda();

    return;
}
//
//#define Mask_width  5
//#define Mask_radius Mask_width/2
//#define TILE_WIDTH 16
//#define w (TILE_WIDTH + Mask_width - 1)
//#define clamp(x) (min(max((x), 0.0), 1.0))
//// 2D version (?)
//__global__ void convolution(float *I, const float* __restrict__ M, float *P,
//                            int channels, int width, int height) {
//    __shared__ float N_ds[w][w];
//    int k;
//    for (k = 0; k < channels; k++) {
//        // First batch loading
//        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
//                destY = dest / w, destX = dest % w,
//                srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius,
//                srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
//                src = (srcY * width + srcX) * channels + k;
//        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
//            N_ds[destY][destX] = I[src];
//        else
//            N_ds[destY][destX] = 0;
//
//        // Second batch loading
//        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
//        destY = dest / w, destX = dest % w;
//        srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
//        srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
//        src = (srcY * width + srcX) * channels + k;
//        if (destY < w) {
//            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
//                N_ds[destY][destX] = I[src];
//            else
//                N_ds[destY][destX] = 0;
//        }
//        __syncthreads();
//
//        float accum = 0;
//        int y, x;
//        for (y = 0; y < Mask_width; y++)
//            for (x = 0; x < Mask_width; x++)
//                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
//        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
//        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
//        if (y < height && x < width)
//            P[(y * width + x) * channels + k] = clamp(accum);
//        __syncthreads();
//    }
//}

#define     MASK_WIDTH      3
#define     MASK_RADIUS     MASK_WIDTH / 2
#define     TILE_WIDTH      8
#define         W           (TILE_WIDTH + MASK_WIDTH - 1)

/**
 * GPU 3D Convolution using shared memory
 */
 template <typename ImgType>
__global__ void convolution(ImgType *I, ImgType* M, ImgType *P, int width, int height, int depth)
{
    /***** WRITE TO SHARED MEMORY *****/
    __shared__ ImgType N_ds[W][W][W];

    // First batch loading
    int dest = threadIdx.x + (threadIdx.y * TILE_WIDTH) + (threadIdx.z * TILE_WIDTH * TILE_WIDTH);
    int destTmp = dest;
    int destX = destTmp % W;
    destTmp = destTmp / W;
    int destY = destTmp % W;
    destTmp = destTmp / W;
    int destZ = destTmp;

    int srcZ = destZ + (blockIdx.z * TILE_WIDTH) - MASK_RADIUS;
    int srcY = destY + (blockIdx.y * TILE_WIDTH) - MASK_RADIUS;
    int srcX = destX + (blockIdx.x * TILE_WIDTH) - MASK_RADIUS;
    int src = srcX + (srcY * width) + (srcZ * width * height);

    if(srcZ >= 0 && srcZ < depth && srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destZ][destY][destX] = I[src];
    else
        N_ds[destZ][destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.x + (threadIdx.y * TILE_WIDTH) + (threadIdx.z * TILE_WIDTH * TILE_WIDTH) + TILE_WIDTH * TILE_WIDTH * TILE_WIDTH;

    destTmp = dest;
    destX = destTmp % W;
    destTmp = destTmp / W;
    destY = destTmp % W;
    destTmp = destTmp / W;
    destZ = destTmp;

    srcZ = destZ + (blockIdx.z * TILE_WIDTH) - MASK_RADIUS;
    srcY = destY + (blockIdx.y * TILE_WIDTH) - MASK_RADIUS;
    srcX = destX + (blockIdx.x * TILE_WIDTH) - MASK_RADIUS;
    src = srcX + (srcY * width) + (srcZ * width * height);

    if(destZ < W)
    {
        if(srcZ >= 0 && srcZ < depth && srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destZ][destY][destX] = I[src];
        else
            N_ds[destZ][destY][destX] = 0;
    }
    __syncthreads();

    /***** Perform Convolution *****/
    ImgType sum = 0;
    int z;
    int y;
    int x;
    for(z = 0; z < MASK_WIDTH; z++)
        for(y = 0; y < MASK_WIDTH; y++)
            for(x = 0; x < MASK_WIDTH; x++)
                sum = sum + N_ds[threadIdx.z + z][threadIdx.y + y][threadIdx.x + x] * M[x + (y * MASK_WIDTH) + (z * MASK_WIDTH * MASK_WIDTH)];
    z = threadIdx.z + (blockIdx.z * TILE_WIDTH);
    y = threadIdx.y + (blockIdx.y * TILE_WIDTH);
    x = threadIdx.x + (blockIdx.x * TILE_WIDTH);
    if(z < depth && y < height && x < width)
        P[x + (y * width) + (z * width * height)] = sum;

    __syncthreads();

}

template <typename T>
void compute3rdPartyConv(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &kernel) {
    std::cout << "HELLO" << std::endl;

    T *dInput;
    size_t dataSize = in.mesh.size() * sizeof(T);
    cudaMalloc(&dInput, dataSize);
    cudaMemcpy(dInput, in.mesh.get(), dataSize, cudaMemcpyHostToDevice);
    T *dOutput;
    cudaMalloc(&dOutput, dataSize);
    T *dKernel;
    size_t kernelSize = kernel.mesh.size() * sizeof(T);
    cudaMalloc(&dKernel, kernelSize);
    cudaMemcpy(dKernel, kernel.mesh.get(), kernelSize, cudaMemcpyHostToDevice);

    float mask[] =
            {
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,

                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,

                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f
            };

    T *deviceMaskData;
    cudaMalloc((void **)&deviceMaskData,        MASK_WIDTH  * MASK_WIDTH   * MASK_WIDTH  * sizeof(T));
    cudaMemcpy(deviceMaskData,       mask, MASK_WIDTH  * MASK_WIDTH   * MASK_WIDTH  * sizeof(T), cudaMemcpyHostToDevice);

//    dim3 threadsPerBlock(1, 32, 32);
//    dim3 numBlocks(1,
//                   (in.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
//                   (in.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
//    conv <<<numBlocks, threadsPerBlock>>> (dInput, dOutput, in.x_num, in.y_num, in.z_num, dKernel, kernel.x_num);
    int image_width = in.y_num;
    int image_height = in.x_num;
    int image_depth = in.z_num;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((image_width + TILE_WIDTH - 1) / TILE_WIDTH, (image_height + TILE_WIDTH - 1) / TILE_WIDTH, (image_depth + TILE_WIDTH - 1) / TILE_WIDTH);
    APRTimer timer(true);
    timer.start_timer("DEVICE time");
    convolution<<<dimGrid, dimBlock>>>(dInput, deviceMaskData, dOutput, image_width, image_height, image_depth);
    waitForCuda();
    timer.stop_timer();
    cudaMemcpy(out.mesh.get(), dOutput, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceMaskData);

    return;
}

// SOME CODE FROM
// https://stackoverflow.com/questions/22577857/3d-convolution-with-cuda-using-shared-memory
// for comparison

int test() {

    int image_width  = 16;
    int image_height = 16;
    int image_depth  = 5;

    float *deviceInputImageData;
    float *deviceOutputImageData;
    float *deviceMaskData;

    float data[] =
            {
                    1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                    1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                    1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                    1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                    1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
            };

    float mask[] =
            {
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,

                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,

                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f
            };

    // CHECK CHECK CHECK CHECK CHECK
    int shared_memory_size = W * W * W;
    int block_size = TILE_WIDTH * TILE_WIDTH * TILE_WIDTH;
    int max_size = 3 * block_size;
    std::cout << "Block Size: " << block_size << " - Shared Memory Size: " << shared_memory_size << " - Max Size: " << max_size << std::endl;
    std::cout << "SHARED MEMORY SIZE HAS TO BE SMALLER THAN MAX SIZE IN ORDER TO WORK PROPERLY !!!!!!!";

    cudaMalloc((void **)&deviceInputImageData,  image_width * image_height * image_depth * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData, image_width * image_height * image_depth * sizeof(float));
    cudaMalloc((void **)&deviceMaskData,        MASK_WIDTH  * MASK_WIDTH   * MASK_WIDTH  * sizeof(float));

    cudaMemcpy(deviceInputImageData, data, image_width * image_height * image_depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,       mask, MASK_WIDTH  * MASK_WIDTH   * MASK_WIDTH  * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((image_width + TILE_WIDTH - 1) / TILE_WIDTH, (image_height + TILE_WIDTH - 1) / TILE_WIDTH, (image_depth + TILE_WIDTH - 1) / TILE_WIDTH);
    convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, image_width, image_height, image_depth);
    cudaDeviceSynchronize();

    cudaMemcpy(data, deviceOutputImageData, image_width * image_height * image_depth * sizeof(float), cudaMemcpyDeviceToHost);

    // Print data
    for(int i = 0; i < image_width * image_height * image_depth; ++i)
    {
        if((i % image_width) == 0)
            std::cout << std::endl;

        if((i % (image_width * image_height)) == 0)
            std::cout << std::endl;

        std::cout << data[i] << " - ";
    }

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    return 0;
}