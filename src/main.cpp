
// Needed for APRTimer and AnalysisData
#ifndef APR_BENCHMARK
#define APR_BENCHMARK
#endif

#include "misc/APRTimer.hpp"
#include "AnalysisData.hpp"

#include <gtest/gtest.h>
#include "TestTools.hpp"

#include "data_structures/Mesh/PixelData.hpp"

#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"

#include "algorithm/LocalIntensityScale.hpp"
#include "algorithm/LocalIntensityScaleCuda.h"


#ifdef HAVE_OPENMP
#include <omp.h>
#endif

AnalysisData ad;

namespace {


    TEST(BenchmarkBsplineTest, FULL) {
        ad.file_name = "BenchmarkBsplineTest";

        using ImgType = float;
        const int numOfRepetitions = 5; // how many runs per each mesh size should be done
        const int numOfRepetitionsToSkip = 1; // how many first runs should be skipped from plot generation
        ad.add_float_data("numOfRepetitionsToSkip", numOfRepetitionsToSkip);
        ad.add_float_data("numOfRepetitions", numOfRepetitions);

        ad.add_string_data("xTitle", "Mesh size in GB");
        ad.add_string_data("yTitle", "Time in seconds");
        ad.add_float_data("xNormalizer", 1024*1024*1024);
        ad.add_float_data("numberOfDecimalPointsX", 1);
        ad.add_string_data("plotTitle", "recursive filter Titan X vs 10 x Xeon(R) @2.60GHz");
        
        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        { // must be in scope of block to destroy APRTimer (destructors writes data to AnalysisData)

            APRTimer timer(true);
	    #ifdef HAVE_OPENMP

	    int numOfThreads = omp_get_num_procs();
	    std::cout << "OpenMP reports " << numOfThreads << " threads available." << std::endl;
	    #endif

            unsigned int numberOfTests = 15 + 1;
            for (size_t d = 0; d < numberOfTests; ++d) {
                size_t yLen = 1024;
                size_t zLen = 1024;
                size_t xStartValue = 128;
                size_t deltaStep = 128;
                size_t xLen = xStartValue + deltaStep * d;

                std::cout << "\n\n========================= " << d+1 << "/" << numberOfTests << " ===============\n";
                ad.add_float_data("ticksValue", xLen * zLen * yLen * sizeof(ImgType));
                PixelData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen);
                std::cout << "MESH: " << m << std::endl;
                PixelData<ImgType> mCpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION CPU: " << i + 1 << "/" << numOfRepetitions << "\n";
                    // Calculate bspline on CPU
                    ComputeGradient cg;
                    timer.start_timer("CpuTime");
                    cg.bspline_filt_rec_y(mCpu, lambda, tolerance);
                    cg.bspline_filt_rec_x(mCpu, lambda, tolerance);
                    cg.bspline_filt_rec_z(mCpu, lambda, tolerance);
                    timer.stop_timer();
                }

                PixelData<ImgType> mGpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION GPU: " << i + 1 << "/" << numOfRepetitions << "\n";

                    // Calculate bspline on GPU
                    timer.start_timer("GpuTimeWithMemTransfer");
                    cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_ALL_DIR);
                    timer.stop_timer();
                }
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.05), 0);
            }
        }

        ad.write_analysis_data_hdf5();
    }

    TEST(BenchmarkBsplineTest, DIFFERENT_FILTER_SIZE) {
        ad.file_name = "BenchmarkBsplineVsK0sizeTest";

        using ImgType = float;
        const int numOfRepetitions = 5; // how many runs per each mesh size should be done
        const int numOfRepetitionsToSkip = 1; // how many first runs should be skipped from plot generation
        ad.add_float_data("numOfRepetitionsToSkip", numOfRepetitionsToSkip);
        ad.add_float_data("numOfRepetitions", numOfRepetitions);

        ad.add_string_data("xTitle", "Mesh size in GB");
        ad.add_string_data("yTitle", "Time in seconds");
        ad.add_float_data("xNormalizer", 1);
        ad.add_float_data("numberOfDecimalPointsX", 0);
        ad.add_string_data("plotTitle", "recursive filter Titan X vs 10 x Xeon(R) @2.60GHz");

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.000001;

        { // must be in scope of block to destroy APRTimer (destructors writes data to AnalysisData)

            APRTimer timer(true);
#ifdef HAVE_OPENMP

            int numOfThreads = omp_get_num_procs();
	    std::cout << "OpenMP reports " << numOfThreads << " threads available." << std::endl;
#endif

            unsigned int numberOfTests = 32 + 1;
            for (size_t d = 0; d < numberOfTests; ++d) {
                size_t yLen = 256;
                size_t zLen = 256;
                size_t xLen = 512;
                int k0StartValue = 1;
                int deltaStep = 1;
                int k0 = k0StartValue + deltaStep * d;

                std::cout << "\n\n========================= " << d+1 << "/" << numberOfTests << " ===============\n";
                ad.add_float_data("ticksValue", k0);
                PixelData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen);
                std::cout << "MESH: " << m << std::endl;
                PixelData<ImgType> mCpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION CPU: " << i + 1 << "/" << numOfRepetitions << "\n";
                    // Calculate bspline on CPU
                    ComputeGradient cg;
                    timer.start_timer("CpuTime");
                    cg.bspline_filt_rec_y(mCpu, lambda, tolerance, k0);
                    cg.bspline_filt_rec_x(mCpu, lambda, tolerance, k0);
                    cg.bspline_filt_rec_z(mCpu, lambda, tolerance, k0);
                    timer.stop_timer();
                }

                PixelData<ImgType> mGpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION GPU: " << i + 1 << "/" << numOfRepetitions << "\n";

                    // Calculate bspline on GPU
                    timer.start_timer("GpuTimeWithMemTransfer");
                    cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_ALL_DIR, k0);
                    timer.stop_timer();
                }
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.05), 0);
            }
        }

        ad.write_analysis_data_hdf5();
    }

    TEST(BenchmarkLocalIntensityScaleTest, FULL_OFFSET_2) {
        ad.file_name = "BenchmarkLocalIntensityScaleTestOffset2";

        using ImgType = float;
        const int numOfRepetitions = 5; // how many runs per each mesh size should be done
        const int numOfRepetitionsToSkip = 1; // how many first runs should be skipped from plot generation
        ad.add_float_data("numOfRepetitionsToSkip", numOfRepetitionsToSkip);
        ad.add_float_data("numOfRepetitions", numOfRepetitions);

        ad.add_string_data("xTitle", "Mesh size in GB");
        ad.add_string_data("yTitle", "Time in seconds");
        ad.add_float_data("xNormalizer", 1024*1024*1024);
        ad.add_float_data("numberOfDecimalPointsX", 1);
        ad.add_string_data("plotTitle", "Local Instensity Scale off=2 Titan X vs 10 x Xeon(R) @2.60GHz");

        // Filter parameters
        const int offset = 2;

        { // must be in scope of block to destroy APRTimer (destructors writes data to AnalysisData)

            APRTimer timer(true);
#ifdef HAVE_OPENMP

            int numOfThreads = omp_get_num_procs();
	    std::cout << "OpenMP reports " << numOfThreads << " threads available." << std::endl;
#endif

            unsigned int numberOfTests = 15 + 1;
            for (size_t d = 0; d < numberOfTests; ++d) {
                size_t yLen = 1024;
                size_t zLen = 1024;
                size_t xStartValue = 128;
                size_t deltaStep = 128;
                size_t xLen = xStartValue + deltaStep * d;

                std::cout << "\n\n========================= " << d+1 << "/" << numberOfTests << " ===============\n";
                ad.add_float_data("ticksValue", xLen * zLen * yLen * sizeof(ImgType));
                PixelData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen);
                std::cout << "MESH: " << m << std::endl;

                PixelData<ImgType> mCpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION CPU: " << i + 1 << "/" << numOfRepetitions << "\n";
                    // Calculate bspline on CPU
                    LocalIntensityScale lis;
                    timer.start_timer("CpuTime");
                    lis.calc_sat_mean_y(mCpu, offset);
                    lis.calc_sat_mean_x(mCpu, offset);
                    lis.calc_sat_mean_z(mCpu, offset);
                    timer.stop_timer();
                }

                PixelData<ImgType> mGpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION GPU: " << i + 1 << "/" << numOfRepetitions << "\n";

                    // Calculate bspline on GPU
                    timer.start_timer("GpuTimeWithMemTransfer");
                    calcMean(mGpu, offset);
                    timer.stop_timer();
                }

                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.05), 0);
            }
        }

        ad.write_analysis_data_hdf5();
    }

    TEST(BenchmarkLocalIntensityScaleTest, FULL_OFFSET_6) {
        ad.file_name = "BenchmarkLocalIntensityScaleTestOffset6";

        using ImgType = float;
        const int numOfRepetitions = 5; // how many runs per each mesh size should be done
        const int numOfRepetitionsToSkip = 1; // how many first runs should be skipped from plot generation
        ad.add_float_data("numOfRepetitionsToSkip", numOfRepetitionsToSkip);
        ad.add_float_data("numOfRepetitions", numOfRepetitions);

        ad.add_string_data("xTitle", "Mesh size in GB");
        ad.add_string_data("yTitle", "Time in seconds");
        ad.add_float_data("xNormalizer", 1024*1024*1024);
        ad.add_float_data("numberOfDecimalPointsX", 1);
        ad.add_string_data("plotTitle", "Local Instensity Scale off=6 Titan X vs 10 x Xeon(R) @2.60GHz");

        // Filter parameters
        const int offset = 6;

        { // must be in scope of block to destroy APRTimer (destructors writes data to AnalysisData)

            APRTimer timer(true);
#ifdef HAVE_OPENMP

            int numOfThreads = omp_get_num_procs();
	    std::cout << "OpenMP reports " << numOfThreads << " threads available." << std::endl;
#endif

            unsigned int numberOfTests = 15 + 1;
            for (size_t d = 0; d < numberOfTests; ++d) {
                size_t yLen = 1024;
                size_t zLen = 1024;
                size_t xStartValue = 128;
                size_t deltaStep = 128;
                size_t xLen = xStartValue + deltaStep * d;

                std::cout << "\n\n========================= " << d+1 << "/" << numberOfTests << " ===============\n";
                ad.add_float_data("ticksValue", xLen * zLen * yLen * sizeof(ImgType));
                PixelData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen);
                std::cout << "MESH: " << m << std::endl;

                PixelData<ImgType> mCpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION CPU: " << i + 1 << "/" << numOfRepetitions << "\n";
                    // Calculate bspline on CPU
                    LocalIntensityScale lis;
                    timer.start_timer("CpuTime");
                    lis.calc_sat_mean_y(mCpu, offset);
                    lis.calc_sat_mean_x(mCpu, offset);
                    lis.calc_sat_mean_z(mCpu, offset);
                    timer.stop_timer();
                }

                PixelData<ImgType> mGpu(m, true);
                for (int i = 0; i < numOfRepetitions; ++i) {
                    std::cout << "<<<<<<<<<<<<<<<<<<< REPETITION GPU: " << i + 1 << "/" << numOfRepetitions << "\n";

                    // Calculate bspline on GPU
                    timer.start_timer("GpuTimeWithMemTransfer");
                    calcMean(mGpu, offset);
                    timer.stop_timer();
                }

                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.05), 0);
            }
        }

        ad.write_analysis_data_hdf5();
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
