#include "memBenchmarks.h"

#include "misc/CudaTools.hpp"


namespace {
    void foo() {
        MeshData<float> f(0, 0, 0);
        MeshData<uint16_t > u16(0, 0, 0);
        MeshData<uint8_t > u8(0, 0, 0);
        memCopy2D(f, f);
        memCopy1D(f, f);
        conv2d(f, f, f);
//        conv2d(u16, u16, u16);
//        conv2d(u8, u8, u8);
        conv3d(f, f, f);
        conv3d(u16,u16,u16);
    }

    template<typename T>
    __global__ void memCopy1dKernel(T *in, T *out, size_t len) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < len) {
            out[idx] = in[idx];
        }
    }

    template<typename T>
    __global__ void memCopy1dKernelB(T *in, T *out, size_t len, int copyPerBlock) {
        int idx = (blockIdx.x * copyPerBlock) * blockDim.x + threadIdx.x;
        for (int i = 0; i < copyPerBlock; ++i) {
//            if (idx < len) {
                out[idx] = in[idx];
                idx += blockDim.x;
//            }
        }
    }

    template<typename T>
    __global__ void memCopy2dKernel(T *in, T *out, int xLen, int yLen) {
        int xi = blockIdx.x * blockDim.x + threadIdx.x;
        int yi = blockIdx.y * blockDim.y + threadIdx.y;
        if (xi < xLen && yi < yLen) {
            out[yi * xLen + xi] = in[yi * xLen + xi];
        }
    }

    template<typename T>
    __global__ void memCopy2dTestKernel(T *in, T *out, int xLen, int yLen) {
        int xi = blockIdx.x * blockDim.x + threadIdx.x;
        if (xi < xLen) {
            size_t idx = xi;
            for (int y = 0; y < yLen; ++y) {
                out[idx] = in[idx];
                idx += xLen;
            }
        }
    }

    void printThroughput(const APRTimer &timer, size_t dataSize, const int numOfRepetitions) {
        double t = timer.timings.back() / numOfRepetitions;
        const size_t gigaByte = 1000 * 1000 * 1000;
        // 2* since once read and once written
        std::cout << "Data throughput: " << (double) 2 * dataSize / t / gigaByte << " GB/s in time " << t << std::endl;
    }
}


template <typename T>
void memCopy1D(const MeshData<T> &in, MeshData<T> &out) {
    APRTimer timer(true);

    T *dInput;
    size_t dataSize = in.mesh.size() * sizeof(T);
    cudaMalloc(&dInput, dataSize);
    cudaMemcpy(dInput, in.mesh.get(), dataSize, cudaMemcpyHostToDevice);

    T *dOutput;
    cudaMalloc(&dOutput, dataSize);

    const int numOfRepetitions = 5;
    {
        dim3 threadsPerBlock(64);
        dim3 numBlocks((in.mesh.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
        printCudaDims(threadsPerBlock, numBlocks);

        timer.start_timer("MEM_COPY");
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy1dKernel <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, in.mesh.size());
            waitForCuda();
        }
        timer.stop_timer();
        printThroughput(timer, dataSize, numOfRepetitions);
    }
    {
        dim3 threadsPerBlock(64);
        int copyPerBlock = 2;
        dim3 numBlocks(((in.mesh.size() + threadsPerBlock.x - 1) / threadsPerBlock.x + copyPerBlock - 1) / copyPerBlock);


        printCudaDims(threadsPerBlock, numBlocks);

        timer.start_timer("MEM_COPY2");
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy1dKernelB <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, in.mesh.size(), copyPerBlock);
            waitForCuda();
        }
        timer.stop_timer();
        printThroughput(timer, dataSize, numOfRepetitions);
    }
    getDataFromKernel(out, dataSize, dOutput);

    return;
}

template <typename T>
void memCopy2D(const MeshData<T> &in, MeshData<T> &out) {
    APRTimer timer(true);

    T *dInput;
    size_t dataSize = in.mesh.size() * sizeof(T);
    cudaMalloc(&dInput, dataSize);
    cudaMemcpy(dInput, in.mesh.get(), dataSize, cudaMemcpyHostToDevice);

    T *dOutput;
    cudaMalloc(&dOutput, dataSize);

    int xLen = in.y_num;
    int yLen = in.x_num;


    const int numOfRepetitions = 100;
    {
        dim3 threadsPerBlock(64, 1);
        dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (yLen + threadsPerBlock.y - 1) / threadsPerBlock.y);
        printCudaDims(threadsPerBlock, numBlocks);
        timer.start_timer("MEM_COPY");
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy2dKernel <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, xLen, yLen);
            waitForCuda();
        }
        timer.stop_timer();
        printThroughput(timer, dataSize, numOfRepetitions);
    }
    {
        dim3 threadsPerBlock(64);
        dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x);
        printCudaDims(threadsPerBlock, numBlocks);
        timer.start_timer("MEM_COPY");
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy2dTestKernel <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, xLen, yLen);
            waitForCuda();
        }
        timer.stop_timer();
        printThroughput(timer, dataSize, numOfRepetitions);
    }
    getDataFromKernel(out, dataSize, dOutput);

    return;
}

__constant__  char kernelMemory[27 * 4];

/**
 * Kernel:
 *
 *    a d g
 *    b e h
 *    c f i
 *
 */
template<typename T>
__global__ void conv2dKernel(const T * __restrict__ in, T * __restrict__ out, int xLen, int yLen) {
    // xi can be in range [-1, xLen]
    int xi = (blockIdx.x * blockDim.x + threadIdx.x)- blockIdx.x * 2 - 1;
    int origXI = xi;
    // boundary check
    if (xi > xLen) return;
    bool boundary = (xi == -1) || (xi == xLen);

    const T * kernel = (T*)kernelMemory;
    const unsigned int active = __activemask();
    const int workerIdx = threadIdx.x;

    // Boundary handling (repeat boundary element)
    if (xi < 0) xi = 0;
    if (xi == xLen) xi = xLen - 1;

    T m1 = 0, m2 = 0, m3 = 0;
    T l1 = 0, l2 = 0, l3 = 0;
    T r1 = 0, r2 = 0, r3 = 0;
    size_t offset =  xi;
    size_t writeOffset = xi - xLen;
    int k = 0;
    bool firstElement = false;
    for (int yi = -1; yi <= yLen; ++yi) {
        T v = in[offset];

        switch(k) {
            case 0: {
                m1 = 0;
                m1 += v * kernel[1];
                m2 += v * kernel[7];
                m3 += v * kernel[4];

                r1 = 0;
                r1 += v * kernel[2];
                r2 += v * kernel[8];
                r3 += v * kernel[5];

                l1 = 0;
                l1 += v * kernel[0];
                l2 += v * kernel[6];
                l3 += v * kernel[3];

                T left = __shfl_sync(active, l2, workerIdx + blockDim.x - 1, blockDim.x);
                T right = __shfl_sync(active, r2, workerIdx + 1, blockDim.x);
//                printf("(%d, %d) %d m2=%f %f %f\n", origXI, yi, writeOffset, left, m2, right);
                if (firstElement && !boundary) out[writeOffset] = m2 + left + right;

                break;
            }
            case 1: {
                m2 = 0;
                m1 += v * kernel[4];
                m2 += v * kernel[1];
                m3 += v * kernel[7];

                r2 = 0;
                r1 += v * kernel[5];
                r2 += v * kernel[2];
                r3 += v * kernel[8];

                l2 = 0;
                l1 += v * kernel[3];
                l2 += v * kernel[0];
                l3 += v * kernel[6];

                T left = __shfl_sync(active, l3, workerIdx + blockDim.x - 1, blockDim.x);
                T right = __shfl_sync(active, r3, workerIdx + 1, blockDim.x);
//                printf("(%d, %d) %d m3=%f\n", origXI, yi, writeOffset, m3);
                if (firstElement && !boundary) out[writeOffset] = m3 + left + right;
                break;
            }
            case 2: {
                m3 = 0;
                m1 += v * kernel[7];
                m2 += v * kernel[4];
                m3 += v * kernel[1];

                r3 = 0;
                r1 += v * kernel[8];
                r2 += v * kernel[5];
                r3 += v * kernel[2];

                l3 = 0;
                l1 += v * kernel[6];
                l2 += v * kernel[3];
                l3 += v * kernel[0];

                firstElement = true;

                T left = __shfl_sync(active, l1, workerIdx + blockDim.x - 1, blockDim.x);
                T right = __shfl_sync(active, r1, workerIdx + 1, blockDim.x);
//                printf("(%d, %d) %d m1=%f\n", origXI, yi, writeOffset, m1);
                if (firstElement && !boundary) out[writeOffset] = m1 + left + right;

                break;
            }
        }
        k = (k + 1) % 3;

//        out[offset] = v;

//        printf("\n");
//        printf("(%d, %d) = (%f, %f, %f)\n",  origXI, yi, prevElement, v, nextElement);

        if (yi >= 0 && yi < yLen) {
            if (yi < yLen - 1) offset += xLen;
            writeOffset += xLen;
        }
    }
}

template <typename T>
void conv2d(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &aKernel) {
    APRTimer timer(true);

    T *dInput;
    size_t dataSize = in.mesh.size() * sizeof(T);
    cudaMalloc(&dInput, dataSize);
    cudaMemcpy(dInput, in.mesh.get(), dataSize, cudaMemcpyHostToDevice);

    T *dOutput;
    cudaMalloc(&dOutput, dataSize);

    cudaMemcpyToSymbol(kernelMemory, aKernel.mesh.get(), aKernel.mesh.size() * sizeof(T));


    int xLen = in.y_num;
    int yLen = in.x_num;

    const int numOfRepetitions = 15;
    const int numOfWorkers = 32;
    const int numOfPointsProcessed = numOfWorkers - 2; // -2 boundary points
    dim3 threadsPerBlock(numOfWorkers);

    dim3 numBlocks((xLen + numOfPointsProcessed - 1) / numOfPointsProcessed);
    printCudaDims(threadsPerBlock, numBlocks);

    timer.start_timer("CONV");
    for (int r = 0; r < numOfRepetitions; ++r) {
        conv2dKernel <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, xLen, yLen);
        waitForCuda();
    }
    timer.stop_timer();
    printThroughput(timer, dataSize, numOfRepetitions);

    getDataFromKernel(out, dataSize, dOutput);

    return;
}

template<typename T>
__global__ void conv3dKernel2(const T * __restrict__ in, T * __restrict__ out, int xLen, int yLen, int zLen) {
    // xi can be in range [-1, xLen]
    int xi = (blockIdx.x * blockDim.x + threadIdx.x);// - blockIdx.x * 2 - 1;
    int zi = (blockIdx.z * blockDim.z + threadIdx.z);// - blockIdx.z * 2 - 1;

    int origXI = xi;
    int origZI = zi;

    // boundary check
    if (xi > xLen || zi > zLen) return;
    bool boundary = (xi == -1) || (xi == xLen) || (zi == -1) || (zi == zLen);

    const T * kernel = (T*)kernelMemory;
    const unsigned int active = __activemask();
    const int workerIdx = threadIdx.x;

    // Boundary handling (repeat boundary element)
    if (xi < 0) xi = 0;
    if (xi == xLen) xi = xLen - 1;
    if (zi < 0) zi = 0;
    if (zi == zLen) zi = zLen - 1;

    T m1 = 0, m2 = 0, m3 = 0;
    T l1 = 0, l2 = 0, l3 = 0;
    T r1 = 0, r2 = 0, r3 = 0;

    size_t offset =  xi + zi * xLen*yLen;
    size_t writeOffset = offset - xLen;
    int k = 0;
    bool firstElement = false;
    for (int yi = -1; yi <= yLen; ++yi) {
        T v = in[offset];
        out[offset] = v;

        if (yi >= 0 && yi < yLen) {
            if (yi < yLen - 1) offset += xLen;
            writeOffset += xLen;
        }
    }
}


template<typename T>
__global__ void conv3dKernel(const T * __restrict__ in, T * __restrict__ out, int xLen, int yLen, int zLen) {
    // xi can be in range [-1, xLen]
    int xi = (blockIdx.x * blockDim.x + threadIdx.x);// - blockIdx.x * 2 - 1;
    int yi = (blockIdx.y * blockDim.y + threadIdx.y);// - blockIdx.y * 2 - 1;

    int origXI = xi;
    int origYI = yi;

    // boundary check
    if (xi > xLen || yi > yLen) return;
    bool boundary = (xi == -1) || (xi == xLen) || (yi == -1) || (yi == yLen);

    const T * kernel = (T*)kernelMemory;
    const unsigned int active = __activemask();
    const int workerIdx = threadIdx.x;

    // Boundary handling (repeat boundary element)
    if (xi < 0) xi = 0;
    if (xi == xLen) xi = xLen - 1;
    if (yi < 0) yi = 0;
    if (yi == yLen) yi = yLen - 1;

    T m1 = 0, m2 = 0, m3 = 0;
    T l1 = 0, l2 = 0, l3 = 0;
    T r1 = 0, r2 = 0, r3 = 0;

    size_t offset =  xi + yi * xLen;
    int k = 0;
    bool firstElement = false;
    for (int zi = -1; zi <= zLen; ++zi) {
        T v = in[offset];
        out[offset] = v;

        if (zi >= 0 && zi < zLen) {
            if (zi < zLen - 1) offset += xLen * yLen;
        }
    }
}
template <typename T>
void conv3d(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &aKernel) {
    APRTimer timer(true);

    T *dInput;
    size_t dataSize = in.mesh.size() * sizeof(T);
    cudaMalloc(&dInput, dataSize);
    cudaMemcpy(dInput, in.mesh.get(), dataSize, cudaMemcpyHostToDevice);

    T *dOutput;
    cudaMalloc(&dOutput, dataSize);

    cudaMemcpyToSymbol(kernelMemory, aKernel.mesh.get(), aKernel.mesh.size() * sizeof(T));


    int xLen = in.y_num;
    int yLen = in.x_num;
    int zLen = in.z_num;

    const int numOfRepetitions = 5;
    const int numOfWorkers = 32;
    const int numOfPointsProcessed = numOfWorkers;// - 2; // -2 boundary points
    const int numOfRows = 4;
    const int numOfRowsProcessed = numOfRows;// -2; // -2 boundary points

    {
        dim3 threadsPerBlock(numOfWorkers, numOfRows, 1);
        dim3 numBlocks((xLen + numOfPointsProcessed - 1) / numOfPointsProcessed,
                       (yLen + numOfRowsProcessed - 1) / numOfRowsProcessed, 1);
        printCudaDims(threadsPerBlock, numBlocks);
        timer.start_timer("CONV");
        for (int r = 0; r < numOfRepetitions; ++r) {
            conv3dKernel << < numBlocks, threadsPerBlock >> > (dInput, dOutput, xLen, yLen, zLen);
            waitForCuda();
        }
        timer.stop_timer();
        printThroughput(timer, dataSize, numOfRepetitions);
    }
    {
        dim3 threadsPerBlock(numOfWorkers, 1, numOfRows);
        dim3 numBlocks((xLen + numOfPointsProcessed - 1) / numOfPointsProcessed, 1,
                       (zLen + numOfRowsProcessed - 1) / numOfRowsProcessed);
        printCudaDims(threadsPerBlock, numBlocks);
        timer.start_timer("CONV");
        for (int r = 0; r < numOfRepetitions; ++r) {
            conv3dKernel2 << < numBlocks, threadsPerBlock >> > (dInput, dOutput, xLen, yLen, zLen);
            waitForCuda();
        }
        timer.stop_timer();
        printThroughput(timer, dataSize, numOfRepetitions);
    }
    getDataFromKernel(out, dataSize, dOutput);

    return;
}
