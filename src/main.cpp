#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/LocalIntensityScale.hpp"
#include "algorithm/LocalIntensityScaleCuda.h"
#include "AnalysisData.hpp"
#include "TestTools.hpp"

namespace {
    TEST(LocalIntensityScaleTest, 1D_Y_DIR) {
        {   // OFFSET=0


            APRTimer timer;
            char *name = "cmdName";
            AnalysisData ad("This is name", "And this is description", 1, &name);

            MeshData<float> m(8, 1, 1, 0);
            float dataIn[] = {3, 6, 9, 12, 15, 18, 21, 24};
            initFromZYXarray(m, dataIn);
            float expect[] = {3, 6, 9, 12, 15, 18, 21, 24};

            for (int i = 0; i < 10; ++i) {
                timer.start_timer("Whatever");

                LocalIntensityScale lis;
                lis.calc_sat_mean_y(m, 0);

                timer.stop_timer();
                ad.add_float_data("outputResultValue", 134.5 + i);
            }

            ad.add_timer(timer);
            ad.write_analysis_data_hdf5();


            ASSERT_TRUE(compare(m, expect, 0.05));
        }
	}
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
