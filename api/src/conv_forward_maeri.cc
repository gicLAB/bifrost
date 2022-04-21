// STONNE
#include "include/stonne_linker.h"
#include "Config.h"
#include "STONNEModel.h"

// TVM
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

// Cost function
#include "include/cost.h"
#include "include/tensor_utils.h"

// Stonne variable taxonomy
// -R: Number of flter rows
// -S: Number of filter columns
// -C: Number of filter and input channels
// -K: Number of filters and output channels
// -G: Number of groups
// -N: Number of inputs (Only 1 is supported so far)
// -X: Number of input rows
// -Y: Number of input columns
// -X_: Number of output columns
// -Y_: Number of output columns

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;

        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.conv2d.nchw")
            .set_body([](TVMArgs args, TVMRetValue *ret) {
                std::string path_to_arch_file = args[0];
                int R = args[1];
                int S = args[2];
                int C = args[3];
                int K = args[4];
                int G = args[5];
                int N = args[6];
                int X = args[7];
                int Y = args[8];
                int H_out = args[9];
                int W_out = args[10];
                int strides_x = args[11];
                int strides_y = args[12];
                int pad_x = args[13];
                int pad_y = args[14];
                int dilation_x = args[15];
                int dilation_y = args[16];
                std::string path_to_tile = args[17];
                int sparsity_ratio = args[18];
                bool tune = args[19];
                std::string tuning_name = args[20];
                bool tune_psums = args[21];
                std::string costs_path = args[22];
                bool stats = args[23];
                DLTensor *input = args[24];
                DLTensor *weight = args[25];
                DLTensor *output = args[26];

                //Creating config  to find out if we are going to
                // run a dense or sparse simulation
                Config stonne_config;
                if (path_to_arch_file != "")
                {
                    stonne_config.loadFile(path_to_arch_file);
                }
                // Set output files
                stonne_config.print_stats_enabled = stats;

                // Run different types of convolutions depending architecture
                int cycles;
                if (tune_psums)
                {

                    float *weight_raw = static_cast<float *>(weight->data);
                    float *output_raw = static_cast<float *>(output->data);
                    float *input_raw = static_cast<float *>(input->data);

                    const int ifmap_size = C * ((X + 2 * pad_x) * (Y + 2 * pad_y));
                    const int ofmap_size = K * H_out * W_out; //X_ and Y_ include padding
                    std::cout << "Executing layer " << tuning_name << std::endl;
                    if (path_to_tile == "")
                    {
                        std::cout << "Tile file parameters must be specified" << std::endl;
                        exit(1);
                    }

                    //Loading the tile
                    Tile tile(path_to_tile);

                    float *ifmap_to_send = Transform_Ifmap_Memory_a(input_raw, C, X, Y, pad_x, pad_y);
                    float *filters_to_send = Transform_Filters_Memory_a(weight_raw, K, G, C, R, S);
                    float *ofmap_raw = new float[ofmap_size];

                    //Tile parameters
                    unsigned int T_R = tile.get_T_R();
                    unsigned int T_S = tile.get_T_S();
                    unsigned int T_C = tile.get_T_C();
                    unsigned int T_K = tile.get_T_K();
                    unsigned int T_G = tile.get_T_G();
                    unsigned int T_N = tile.get_T_N();
                    unsigned int T_X_ = tile.get_T_X_();
                    unsigned int T_Y_ = tile.get_T_Y_();

                    //Executing the accelerator
                    Stonne *stonne_instance = new Stonne(stonne_config);
                    stonne_instance->loadDNNLayer(CONV, tuning_name, R, S, C, K, G, N, X + 2 * pad_x, Y + 2 * pad_y, strides_x, (address_t)ifmap_to_send, (address_t)filters_to_send, (address_t)ofmap_raw, CNN_DATAFLOW);
                    stonne_instance->loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_);
                    unsigned int cost = stonne_instance->mem->getPsums();
                    //Deleting objects
                    delete[] ofmap_raw;
                    delete[] ifmap_to_send;
                    delete[] filters_to_send;
                    delete stonne_instance;
                    reportCost(
                        tuning_name,
                        costs_path,
                        cost

                    );
                    return;
                }

                if (stonne_config.sparsitySupportEnabled())
                {
                    // Convert sparsity ratio to %
                    float sparsity_ratio_float = sparsity_ratio / 100;
                    std::cout << "K init :" << K << std::endl;

                    std::string layer_name = "Conv2dLayerSparse";

                    // Calculate im2col output as h0 * w0 * R * S * C
                    int h0 = (X + 2 * pad_x - (dilation_x * (R - 1) + 1)) / strides_x + 1;
                    int w0 = (Y + 2 * pad_y - (dilation_y * (S - 1) + 1)) / strides_y + 1;

                    // Get input and convert to im2col
                    float *input_raw = static_cast<float *>(input->data);
                    float im2col_array[h0 * w0 * R * S * C];
                    float *input_im2col = im2col_array;

                    // Convert weight and output files to be STONNE compatible
                    float *weight_raw = static_cast<float *>(weight->data);
                    float *output_raw = static_cast<float *>(output->data);

                    // Note that since STONNE only supports sparse GEMM operations, we have to
                    // turn the input to im2col format and
                    // run a GEMM operation instead a CONVOLUTION
                    std::cout << "Run im2col" << std::endl;
                    im2col_cpu(
                        input_raw,
                        C,
                        X,
                        Y,
                        R,
                        S,
                        pad_x,
                        pad_y,
                        strides_x,
                        strides_y,
                        dilation_x,
                        dilation_y,
                        input_im2col);

                    // Getting GEMM dimensions
                    int gemm_M = K;
                    int gemm_K = R * S * C;
                    int gemm_N = h0 * w0;

                    cycles = simulateSparseGemmForward(
                        layer_name,
                        input_im2col,
                        weight_raw,
                        output_raw,
                        N,
                        G,
                        gemm_M,
                        gemm_K,
                        gemm_N,
                        sparsity_ratio_float,
                        stonne_config,
                        MK_STA_KN_STR);
                }
                else if (!stonne_config.convOperationSupported())
                {
                    // If CONV itself is not supported,
                    // run it as a GEMM (e.g., the TPU)

                    // Convert weight and output files to be STONNE compatible
                    float *weight_raw = static_cast<float *>(weight->data);
                    float *output_raw = static_cast<float *>(output->data);

                    // Calculate im2col output as h0 * w0 * R * S * C
                    int h0 = (X + 2 * pad_x - (dilation_x * (R - 1) + 1)) / strides_x + 1;
                    int w0 = (Y + 2 * pad_y - (dilation_y * (S - 1) + 1)) / strides_y + 1;

                    // Get input and convert to im2col
                    float *input_raw = static_cast<float *>(input->data);
                    float im2col_array[h0 * w0 * R * S * C];
                    float *input_im2col = im2col_array;
                    im2col_cpu(
                        input_raw,
                        C,
                        X,
                        Y,
                        R,
                        S,
                        pad_x,
                        pad_y,
                        strides_x,
                        strides_y,
                        dilation_x,
                        dilation_y,
                        input_im2col);

                    // Tranpose the result for the TPU
                    float im2col_array_tranposed[h0 * w0 * R * S * C];
                    float *input_im2col_t = im2col_array_tranposed;
                    transpose(input_im2col, input_im2col_t, R * S * C, h0 * w0);

                    // Getting GEMM dimensions
                    int gemm_M = K;
                    int gemm_K = R * S * C;
                    int gemm_N = h0 * w0;
                    cycles = simulateDenseGemmForward("TPU", input_im2col_t, weight_raw, output_raw, N, G, gemm_M, gemm_K, gemm_N, path_to_tile, stonne_config);
                }
                else

                { // Run a dense forward convolution

                    // Cast pointers so they can be fed into stonne
                    float *input_raw = static_cast<float *>(input->data);
                    float *weight_raw = static_cast<float *>(weight->data);
                    float *output_raw = static_cast<float *>(output->data);

                    // Choose name for output statistics
                    std::string layer_name = "Conv2dLayerDense";

                    // Run the simulated forward convolution
                    Stonne *stonne_instance = simulateDenseConvForward(
                        layer_name,
                        input_raw,
                        weight_raw,
                        output_raw,
                        R,
                        S,
                        C,
                        K,
                        G,
                        N,
                        X,
                        Y,
                        H_out,
                        W_out,
                        strides_x,
                        pad_x,
                        pad_y,
                        path_to_tile,
                        stonne_config);

                    cycles = stonne_instance->n_cycles;
                    delete stonne_instance;
                }
                if (tune)
                // If the hardware is being tuned, report the cost
                {
                    reportCost(
                        tuning_name,
                        costs_path,
                        cycles

                    );
                }
                else
                {
                    reportTotalCycles(
                        tuning_name,
                        "bifrost_temp/cycles.json",
                        cycles);
                }
            });

    } // namespace contrib
} // namespace tvm




