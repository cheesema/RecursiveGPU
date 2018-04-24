#include "memBenchmarks.h"

#include "misc/CudaTools.hpp"


namespace {
    void foo() {
        MeshData<float> f(0, 0, 0);
        MeshData<uint16_t > u16(0, 0, 0);
        memCopy2D(f, f);
        memCopy1D(f, f);
    }
    template<typename T>
    __global__ void memCopy1dKernel(T *in, T *out, size_t len) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < len) {
            out[idx] = in[idx];
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

    dim3 threadsPerBlock(64);
    dim3 numBlocks((in.mesh.size() + threadsPerBlock.x - 1) / threadsPerBlock.x );
    printCudaDims(threadsPerBlock, numBlocks);

    const int numOfRepetitions = 100;
    timer.start_timer("MEM_COPY");
    for (int i = 0; i < numOfRepetitions; ++i) {
        memCopy1dKernel <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, in.mesh.size());
        waitForCuda();
    }
    timer.stop_timer();

    printThroughput(timer, dataSize, numOfRepetitions);

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
            memCopy2dKernel << < numBlocks, threadsPerBlock >> > (dInput, dOutput, xLen, yLen);
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
            memCopy2dTestKernel << < numBlocks, threadsPerBlock >> > (dInput, dOutput, xLen, yLen);
            waitForCuda();
        }
        timer.stop_timer();
        printThroughput(timer, dataSize, numOfRepetitions);
    }
    getDataFromKernel(out, dataSize, dOutput);

    return;
}


