// STONNE
#include "include/stonne_linker.h"
#include "Config.h"
#include "STONNEModel.h"

// TVM
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;

        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.matmul")
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
            });

            std::string layer_name
            torch::Tensor input
            torch::Tensor other
            std::string path_to_arch_file
            std::string path_to_tile, 
            float sparsity_level
            
            /*
            * The criteria to carry out the calculation of the N-dimensional matrix multiplication is 
            * explained as follows: 
            *
            * - If both tensors are 1-dimensional, the dot product (scalar) is returned.
            * - If both arguments are 2-dimensional, the matrix-matrix product is returned.
            * - If the first argument is 1-dimensional and the second argument is 2-dimensional,
            *   a 1 is prepended to its dimension for the purpose of the matrix multiply. 
            *   After the matrix multiply, the prepended dimension is removed. 
            * - If the first argument is 2-dimensional and the second argument is 1-dimensional, 
            *   the matrix-vector product is returned.
            *
            * - If both arguments are at least 1-dimensional and at least one argument is 
            *   N-dimensional (where N > 2), then a batched matrix multiply is returned. 
            *   If the first argument is 1-dimensional, a 1 is prepended to its dimension 
            *   for the purpose of the batched matrix multiply and removed after. If the 
            *   second argument is 1-dimensional, a 1 is appended to its dimension for the 
            *   purpose of the batched matrix multiple and removed after. The non-matrix 
            *   (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). 
            *   For example, if input is a (j \times 1 \times n \times m)(j×1×n×m) tensor 
            *   and other is a (k \times m \times p)(k×m×p) tensor, out will be an 
            *   (j \times k \times n \times p)(j×k×n×p) tensor.
            */

            int first_matrix_ndim = input.dim();
            int second_matrix_ndim = other.dim();

            //If both tensors are 1-dimensional, the dot product (scalar) is returned.
            if ((first_matrix_ndim == 1) && (second_matrix_ndim == 1))
            {
                //If both tensors are 1-dimensional, the dot product (scalar) is returned.
                torch::Tensor input_transformed = input.view({1, -1});
                torch::Tensor other_transformed = other.view({-1, 1});
                //torch::Tensor output = torch::mm(input_transformed, other_transformed);
                torch::Tensor output = simulated_linear_forward(layer_name, input_transformed, other_transformed, path_to_arch_file, path_to_tile, sparsity_level, false);
                output = output.view({output.sizes()[1]});
                return output;
            }

            //If both arguments are 2-dimensional, the matrix-matrix product is returned.
            else if ((first_matrix_ndim == 2) && (second_matrix_ndim == 2))
            {
                //return torch::mm(input, other);
                return simulated_linear_forward(layer_name, input, other, path_to_arch_file, path_to_tile, sparsity_level, false);
            }

            //If the first argument is 1-dimensional and the second argument is 2-dimensional,
            //a 1 is prepended to its dimension for the purpose of the matrix multiply.
            //  After the matrix multiply, the prepended dimension is removed.
            else if ((first_matrix_ndim == 1) && (second_matrix_ndim == 2))
            {
                //If both tensors are 1-dimensional, the dot product (scalar) is returned.
                torch::Tensor input_transformed = input.view({1, -1});
                std::cout << "Input dimension: " << input_transformed.sizes() << std::endl;
                std::cout << "Other dimension: " << other.sizes() << std::endl;
                //torch::Tensor output = torch::mm(input_transformed, other);
                torch::Tensor output = simulated_linear_forward(layer_name, input_transformed, other, path_to_arch_file, path_to_tile, sparsity_level, false);
                std::cout << "Output dimension: " << output.sizes() << std::endl;
                output = output.view({output.sizes()[1]});
                return output;
            }
            //If the first argument is 2-dimensional and the second argument is 1-dimensional,
            //the matrix-vector product is returned.
            else if ((first_matrix_ndim == 2) && (second_matrix_ndim == 1))
            {
                //If both tensors are 1-dimensional, the dot product (scalar) is returned.
                torch::Tensor other_transformed = other.view({-1, 1});
                std::cout << "Input dimension: " << input.sizes() << std::endl;
                std::cout << "Other dimension: " << other_transformed.sizes() << std::endl;
                //torch::Tensor output = torch::mm(input, other_transformed);
                torch::Tensor output = simulated_linear_forward(layer_name, input, other_transformed, path_to_arch_file, path_to_tile, sparsity_level, false);
                std::cout << "Output dimension: " << output.sizes() << std::endl;
                output = output.view({output.sizes()[0]});
                return output;
            }

            else
            {
                //Adding one dimension when corresponds if the matrices are not at least 2-dimensional
                torch::Tensor input_transformed = input;
                torch::Tensor other_transformed = other;
                if ((first_matrix_ndim == 1))
                {
                    input_transformed = input.unsqueeze(0);
                    first_matrix_ndim++;
                }

                else if (second_matrix_ndim == 1)
                {
                    other_transformed = other.unsqueeze(1);
                    second_matrix_ndim++;
                }

                if (first_matrix_ndim != second_matrix_ndim)
                { //Adding extra dimensions for broadcasting
                    torch::Tensor longer_matrix = input_transformed;
                    torch::Tensor shorter_matrix = other_transformed;
                    if (second_matrix_ndim > first_matrix_ndim)
                    {
                        longer_matrix = other_transformed;
                        shorter_matrix = input_transformed;
                    }

                    int diff = longer_matrix.dim() - shorter_matrix.dim();
                    for (int i = 0; i < diff; i++)
                    {
                        shorter_matrix = shorter_matrix.unsqueeze(0); //Adding extra dimension to make both equal
                    }

                    input_transformed = longer_matrix;
                    other_transformed = shorter_matrix;
                    if (second_matrix_ndim > first_matrix_ndim)
                    {
                        other_transformed = longer_matrix;
                        input_transformed = shorter_matrix;
                    }
                }

                //At this point both matrices must equal their dimensions
                if (input_transformed.dim() != other_transformed.dim())
                {
                    std::cerr << "The two matrices are not broadcasted" << std::endl;
                    std::cout << "input transformed dimensions: " << input_transformed.dim() << std::endl;
                    std::cout << "other transformed dimensions: " << other_transformed.dim() << std::endl;
                    exit(1);
                }

                //We iterate over every element getting the bath sizes
                std::cout << "First matrix dimensions: " << input_transformed.sizes() << std::endl;
                std::cout << "Second matrix dimensions: " << other_transformed.sizes() << std::endl;

                //Let's check if the dimensions are broadcasted
                for (int i = 0; i < input_transformed.dim() - 2; i++)
                {
                    if ((input_transformed.sizes()[i] != other_transformed.sizes()[i]) && (input_transformed.sizes()[i] != 1) && (other_transformed.sizes()[i] != 1))
                    {
                        std::cerr << "The two matrices are not broadcasted. input[" << input_transformed.sizes()[i] << "] is not compatible with other[" << other_transformed.sizes()[i] << "]" << std::endl;
                        exit(1);
                    }
                }

                //Perform the matrix multiplication
                if (input_transformed.dim() == 3)
                {
                    int max_dim_0 = (input_transformed.sizes()[0] > other_transformed.sizes()[0]) ? input_transformed.sizes()[0] : other_transformed.sizes()[0];
                    torch::Tensor output = torch::rand({max_dim_0, input_transformed.sizes()[1], other_transformed.sizes()[2]});
                    for (int i = 0; i < max_dim_0; i++)
                    {
                        int index_first = (input_transformed.sizes()[0] > 1) ? i : 0;
                        int index_second = (other_transformed.sizes()[0] > 1) ? i : 0;
                        std::cout << "Computing first matrix " << input_transformed[index_first].sizes() << std::endl;
                        std::cout << "Computing second matrix " << other_transformed[index_second].sizes() << std::endl;
                        std::string layer_name_batch = layer_name + "_B_" + std::to_string(i);
                        torch::Tensor curr_output = simulated_linear_forward(layer_name_batch, input_transformed[index_first], other_transformed[index_second], path_to_arch_file, path_to_tile, sparsity_level, false);
                        output.slice(0, i, i + 1) = curr_output;
                    }

                    return output;
                }

                else if (input_transformed.dim() == 4)
                {
                    std::cout << "Mtrix with dimensions 4" << std::endl;
                    int max_dim_0 = (input_transformed.sizes()[0] > other_transformed.sizes()[0]) ? input_transformed.sizes()[0] : other_transformed.sizes()[0];
                    int max_dim_1 = (input_transformed.sizes()[1] > other_transformed.sizes()[1]) ? input_transformed.sizes()[1] : other_transformed.sizes()[1];
                    torch::Tensor output = torch::rand({max_dim_0, max_dim_1, input_transformed.sizes()[2], other_transformed.sizes()[3]});
                    for (int i = 0; i < max_dim_0; i++)
                    {
                        int index_first_0 = (input_transformed.sizes()[0] > 1) ? i : 0;
                        int index_second_0 = (other_transformed.sizes()[0] > 1) ? i : 0;
                        for (int j = 0; j < max_dim_1; j++)
                        {
                            int index_first_1 = (input_transformed.sizes()[1] > 1) ? j : 0;
                            int index_second_1 = (other_transformed.sizes()[1] > 1) ? j : 0;
                            //torch::Tensor curr_output = torch::matmul(input_transformed[index_first_0][index_first_1], other_transformed[index_second_0][index_second_1]);
                            std::string layer_name_batch = layer_name + "_B_" + std::to_string(i) + "_" + std::to_string(j);
                            torch::Tensor curr_output = simulated_linear_forward(layer_name_batch, input_transformed[index_first_0][index_first_1], other_transformed[index_second_0][index_second_1], path_to_arch_file, path_to_tile, sparsity_level, false);
                            output.slice(0, i, i + 1).slice(1, j, j + 1) = curr_output;
                        }
                    }

                    return output;
                }

                else
                {
                    std::cerr << ">5-dimension matrix multiplications not supported" << std::endl;
                    exit(1);
                }

            } // namespace contrib
    }         // namespace tvm
