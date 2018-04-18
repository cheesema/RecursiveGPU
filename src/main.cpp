#include "misc/APRTimer.hpp"
#include "AnalysisData.hpp"

#include <gtest/gtest.h>
#include "TestTools.hpp"

#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeBsplineRecursiveFilterCuda.h"

AnalysisData ad;

namespace {
    TEST(BenchmarkBsplineTest, FULL) {
        ad.file_name = "asdf";

        using ImgType = float;
        const int numOfRepetitions = 5; // how many runs per each mesh size should be done
        const int numOfRepetitionsToSkip = 1; // how many first runs should be skipped from plot generation
        ad.add_float_data("numOfRepetitionsToSkip", numOfRepetitionsToSkip);
        ad.add_float_data("numOfRepetitions", numOfRepetitions);

        ad.add_string_data("xTitle", "Mesh size in GB");
        ad.add_string_data("yTitle", "Time in seconds");
        ad.add_float_data("xNormalizer", 1024*1024*1024);
        ad.add_float_data("numberOfDecimalPointsX", 1);
        ad.add_string_data("plotTitle", "BSpline recursive filter Titan X vs 10 x Xeon(R) CPU E5-2660 v3 @ 2.60GHz");
        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        { // must be in scope of block to destroy APRTimer (destructors writes data to AnalysisData)

            APRTimer timer(true);

            int numberOfTests = 7 + 1;
            for (size_t d = 0; d < numberOfTests; ++d) {
                size_t yLen = 1024;
                size_t zLen = 1024;
                size_t xStartValue = 256;
                size_t deltaStep = 256;
                size_t xLen = xStartValue + deltaStep * d;

                std::cout << "========================= " << d+1 << "/" << numberOfTests << " ===============\n";
                ad.add_float_data("ticksValue", xLen * zLen * yLen * sizeof(ImgType));
                MeshData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen);
                std::cout << "MESH: " << m << std::endl;
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION CPU: " << i + 1 << "/" << numOfRepetitions << "\n";
                    // Calculate bspline on CPU
                    ComputeGradient cg;
                    timer.start_timer("CpuTime");
                    MeshData<ImgType> mCpu(m, true);
                    cg.bspline_filt_rec_y(mCpu, lambda, tolerance);
                    cg.bspline_filt_rec_x(mCpu, lambda, tolerance);
                    cg.bspline_filt_rec_z(mCpu, lambda, tolerance);
                    timer.stop_timer();
                }

                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION GPU: " << i + 1 << "/" << numOfRepetitions << "\n";

                    // Calculate bspline on GPU
                    MeshData<ImgType> mGpu(m, true);
                    timer.start_timer("GpuTimeWithMemTransfer");
                    cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_ALL_DIR);
                    timer.stop_timer();
                }

            }
        }

        ad.write_analysis_data_hdf5();
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
