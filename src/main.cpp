#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/LocalIntensityScale.hpp"
#include "algorithm/LocalIntensityScaleCuda.h"
#include "AnalysisData.hpp"
#include "TestTools.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeBsplineRecursiveFilterCuda.h"

namespace {
    TEST(LocalIntensityScaleTest, 1D_Y_DIR) {
        {   // OFFSET=0
            APRTimer timer(true);
            const char *name = "cmdName";
            AnalysisData ad("This is name", "And this is description", 1, &name);
            ad.name = "asdf";

            using ImgType = float;
            MeshData<ImgType> m = getRandInitializedMesh<ImgType>(512,256,256);

            // Filter parameters
            const float lambda = 3;
            const float tolerance = 0.0001;

            for (int i = 0; i < 10; ++i) {

                // Calculate bspline on CPU
                ComputeGradient cg;
                MeshData<ImgType> mCpu(m, true);
                timer.start_timer("CPU bspline");
                cg.bspline_filt_rec_y(mCpu, lambda, tolerance);
                cg.bspline_filt_rec_x(mCpu, lambda, tolerance);
                cg.bspline_filt_rec_z(mCpu, lambda, tolerance);
                timer.stop_timer();

                // Calculate bspline on GPU
                MeshData<ImgType> mGpu(m, true);
                timer.start_timer("GPU bspline");
                cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_ALL_DIR);
                timer.stop_timer();

                EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
            }

            ad.add_timer(timer);
            ad.write_analysis_data_hdf5();
        }
	}
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
