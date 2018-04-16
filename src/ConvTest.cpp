//
// Created by Krzysztof Gonciarz on 4/15/18.
//

#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "TestTools.hpp"
#include "Conv.h"


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
            test();
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
