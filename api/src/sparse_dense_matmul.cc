// STONNE
#include "include/stonne_linker.h"
#include "Config.h"
#include "STONNEModel.h"

// TVM
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

// JSONCPP
#include "json/json.h"
#include "json/json-forwards.h"

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;
        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.sparse_dense_matmul")
            .set_body([](TVMArgs args, TVMRetValue *ret) {
                std::string path_to_arch_file = args[0];
                int M = args[1]; // Batch size
                int K = args[2]; // Number of input neurons
                int N = args[3]; // Number of output neurons
                std::string path_to_tile = args[4];
                int sparsity_ratio = args[5];
                bool stats = args[6];
                DLTensor *input = args[7];
                DLTensor *weight = args[8];
                DLTensor *output = args[9];

                // Add some way to specify layer names
                std::string layer_name = "Test";

                //Here starts the function
                //Creating config file to find out if we are going to run a dense or sparse simulation
                Config stonne_config;
                if (path_to_arch_file != "")
                {
                    stonne_config.loadFile(path_to_arch_file);
                }
                stonne_config.print_stats_enabled = stats;

                // Cast pointers so they can be fed into stonne
                float *input_raw = static_cast<float *>(input->data);
                float *weight_raw = static_cast<float *>(weight->data);
                float *output_raw = static_cast<float *>(output->data);

                if (stonne_config.sparsitySupportEnabled())
                {
                    // Convert sparsity ratio to %
                    float sparsity_ratio_float = sparsity_ratio / 100;
                    simulateSparseGemmForward(
                        layer_name, 
                        weight_raw, 
                        input_raw, 
                        output_raw,
                        1, 1, M, K, N, 
                        sparsity_ratio_float, 
                        stonne_config,
                        MK_STR_KN_STA);
                }
                else
                {
                    simulateDenseGemmForward(
                        layer_name,
                        weight_raw,
                        input_raw,
                        output_raw,
                        1, 1, M, K, N,
                        path_to_tile,
                        stonne_config);
                }
            });

    } // namespace contrib
} // namespace tvm
