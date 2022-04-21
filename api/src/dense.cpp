// STONNE
#include "include/stonne_linker.h"
#include "Config.h"
#include "STONNEModel.h"

// TVM
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

// Cost function
#include "include/cost.h"

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;
        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.dense")
            .set_body([](TVMArgs args, TVMRetValue *ret) {
                std::string path_to_arch_file = args[0];
                int M = args[1]; // Batch size
                int K = args[2]; // Number of input neurons
                int N = args[3]; // Number of output neurons
                std::string path_to_tile = args[4];
                int sparsity_ratio = args[5];
                bool stats = args[6];
                bool tune = args[7];
                bool tune_psums = args[8];
                std::string tuning_name = args[9];
                std::string costs_path = args[10];
                DLTensor *input = args[11];
                DLTensor *weight = args[12];
                DLTensor *output = args[13];

                // Add some way to specify layer names
                std::string layer_name = "Test";

                // Init tuning cost variable
                int cost;

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
                if (tune_psums)
                {
                    cost = simulateDenseGemmForwardPsums(
                        layer_name,
                        weight_raw,
                        input_raw,
                        output_raw,
                        1, 1, M, K, N,
                        path_to_tile,
                        stonne_config);
                }
                else if (stonne_config.sparsitySupportEnabled())
                {
                    // Convert sparsity ratio to %
                    float sparsity_ratio_float = sparsity_ratio / 100;
                    cost = simulateSparseGemmForward(
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
                { //get cycles
                    cost = simulateDenseGemmForward(
                        layer_name,
                        weight_raw,
                        input_raw,
                        output_raw,
                        1, 1, M, K, N,
                        path_to_tile,
                        stonne_config);
                }
                if (tune)
                // If the hardware is being tuned, report the cost
                {
                    reportCost(
                        tuning_name,
                        costs_path,
                        cost

                    );
                }
                else
                {
                    reportTotalCycles(
                        tuning_name,
                        "bifrost_temp/cycles.json",
                        cost);
                }
            });

    } // namespace contrib
} // namespace tvm
