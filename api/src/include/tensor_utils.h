/**
 * Transpose a matrix
 *
 * @param src pointer to float input array
 * @param dst pointer to float output array
 * @param N integer number of cols
 * @param M integer number of rows
 *
 */
void transpose(float *src, float *dst, const int N, const int M);

/**
 * Run im2col
 *
 * @param data_im A pointer of the float array of the inputs
 * @param data_col A pointer to the output float array.
 * Make sure this is the right size (h0*w0*R*S*C)
 */
void im2col_cpu(const float *data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                float *data_col);

float *Transform_Ifmap_Memory_a(const float *bottom_data, const int C,
                                const int X, const int Y, const int pad_x,
                                const int pad_y);
float *Transform_Ifmap_Memory_b(const float *bottom_data, const int C,
                                const int X, const int Y, const int pad_x,
                                const int pad_y);
float *Transform_Ifmap_Memory_c(const float *bottom_data, const int C,
                                const int X, const int Y, const int pad_x,
                                const int pad_y);

float *Transform_Filters_Memory_a(const float *weights, const int K,
                                  const int G, const int C, const int R,
                                  const int S);
float *Transform_Filters_Memory_b(const float *weights, const int K,
                                  const int G, const int C, const int R,
                                  const int S);

void Transform_Ofmap_Memory_a(const float *ofmap_data, float *top_data,
                              const int K, const int X_, const int Y_);
