//
// Created by Krzysztof Gonciarz on 4/15/18.
//

// Needed for APRTimer and AnalysisData
#ifndef APR_BENCHMARK
#define APR_BENCHMARK
#endif

#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "TestTools.hpp"
#include "Conv.h"
#include "misc/APRTimer.hpp"
#include "AnalysisData.hpp"
#include "memBenchmarks.h"


AnalysisData ad;

namespace {
    TEST(ConvolutionTest, SMALL) {
        {
            using ImgType = float;
            MeshData<ImgType> input = getRandInitializedMesh<ImgType>(3,3,1,1,true);
            input.printMesh(3, 1);
            MeshData<ImgType> result(input, false /* don't copy */);
            MeshData<ImgType> kernel(3,3,3,2);

            computeConv(input, result, kernel);
            std::cout << "=========\n";
        }
    }

    TEST(ConvolutionTest, 3RD_PARTY) {
        {
            APRTimer timer(true);
            using ImgType = uint16_t ;
            MeshData<ImgType> input = getRandInitializedMesh<ImgType>(1024,256,1024,1,true);
            std::cout << input << std::endl;
//            input.printMesh(3, 1);

            MeshData<ImgType> result(input, false /* don't copy */);
            MeshData<ImgType> kernel(3,3,3,2);

            timer.start_timer("GpuTimeWithMemTransfer");
            compute3rdPartyConv(input, result, kernel);
            timer.stop_timer();

//            input.printMesh(3, 1);
//            result.printMesh(3,1);
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
