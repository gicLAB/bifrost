void transpose(float *src, float *dst, const int N, const int M) {
// Tranpose a matrix
#pragma omp parallel for
  for (int n = 0; n < N * M; n++) {
    int i = n / N;
    int j = n % N;
    dst[n] = src[M * j + i];
  }
}

// Inspired from Berkeley Vision's Caffe, modified to suit STONNE
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col_cpu(const float *data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                float *data_col) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// NCHW -> NHWC
float *Transform_Ifmap_Memory_a(const float *bottom_data, const int C,
                                const int X, const int Y, const int pad_x,
                                const int pad_y) {
  const int n_channels = C;
  const int input_y = X;
  const int input_x = Y;

  const int input_y_pad = input_y + 2 * pad_y;
  const int input_x_pad = input_x + 2 * pad_x;
  int size_channel = input_y * input_x;
  int n = n_channels * (input_y_pad * input_x_pad);

  float *data_to_send =
      new float[n]; // Creating piece of memory that will use the simulator
  // Adding y padding

  for (int i = 0; i < n; i++) {
    data_to_send[i] = 0.0;
  }
  for (int i = 0; i < n_channels; i++) {
    for (int y = 0; y < input_y; y++) {
      for (int x = 0; x < input_x; x++) {
        data_to_send[(n_channels * ((y + pad_y) * input_x_pad + x + pad_x)) +
                     i] = bottom_data[i * size_channel + y * (input_x) + x];
      }
    }
  }

  return data_to_send;
}

float *Transform_Ifmap_Memory_b(const float *bottom_data, const int C,
                                const int X, const int Y, const int pad_x,
                                const int pad_y) {
  const int n_channels = C;
  const int input_y = X;
  const int input_x = Y;

  const int input_y_pad = input_y + 2 * pad_y;
  const int input_x_pad = input_x + 2 * pad_x;
  int size_channel = input_y * input_x;
  int n = n_channels * (input_y_pad * input_x_pad);

  float *data_to_send =
      new float[n]; // Creating piece of memory that will use the simulator
  // Adding y padding

  for (int i = 0; i < n; i++) {
    data_to_send[i] = 0.0;
  }
  for (int i = 0; i < n_channels; i++) {
    for (int y = 0; y < input_y; y++) {
      for (int x = 0; x < input_x; x++) {
        data_to_send[i * size_channel + y * (input_x) + x] =
            bottom_data[(n_channels * ((y + pad_y) * input_x_pad + x + pad_x)) +
                        i];
      }
    }
  }

  return data_to_send;
}

float *Transform_Ifmap_Memory_c(const float *bottom_data, const int C,
                                const int X, const int Y, const int pad_x,
                                const int pad_y) {
  const int n_channels = C;
  const int input_y = X;
  const int input_x = Y;

  const int input_y_pad = input_y + 2 * pad_y;
  const int input_x_pad = input_x + 2 * pad_x;
  int size_channel = input_y * input_x;
  int n = n_channels * (input_y_pad * input_x_pad);

  float *data_to_send =
      new float[n]; // Creating piece of memory that will use the simulator
  // Adding y padding

  for (int i = 0; i < n; i++) {
    data_to_send[i] = 0.0;
  }
  for (int i = 0; i < n_channels; i++) {
    for (int y = 0; y < input_y; y++) {
      for (int x = 0; x < input_x; x++) {
        data_to_send[(n_channels * ((y + pad_y) * input_x_pad + x + pad_x)) +
                     i] = bottom_data[(n_channels * (y * input_x + x)) + i];
      }
    }
  }

  return data_to_send;
}

// NHWC->NCHW
void Transform_Ofmap_Memory_a(const float *ofmap_data, float *top_data,
                              const int K, const int X_, const int Y_) {
  const int n_channels = K; // n_filters
  const int output_y = X_;
  const int output_x = Y_;

  int size_channel = output_y * output_x;
  int n = n_channels * size_channel;
  for (int i = 0; i < n_channels; i++) {
    for (int y = 0; y < output_y; y++) {
      for (int x = 0; x < output_x; x++) {
        // data_to_send[(n_channels*(y*input_x+x)) +
        // i]=bottom_data[i*size_channel + y*input_x + x];
        top_data[i * size_channel + y * output_x + x] =
            ofmap_data[(n_channels * (y * output_x + x)) +
                       i]; // Filling top_data
      }
    }
  }
}

// KCRS -> RSCK
float *Transform_Filters_Memory_a(const float *weights, const int K,
                                  const int G, const int C, const int R,
                                  const int S) {

  const int n_channels = C / G;
  const int kernel_y = R;
  const int kernel_x = S;
  const int n_filters = K; // this->num_output_;
  int size_channel = kernel_y * kernel_x;
  int size_filter = size_channel * n_channels;
  int n = size_filter * n_filters;

  float *filters_to_send =
      new float[n]; // Creating piece of memory that will use the simulator
  for (int n_f = 0; n_f < n_filters; n_f++) {
    for (int i = 0; i < n_channels; i++) {
      for (int y = 0; y < kernel_y; y++) {
        for (int x = 0; x < kernel_x; x++) {
          filters_to_send[n_f * size_filter +
                          (n_channels * (y * kernel_x + x)) + i] =
              weights[n_f * size_filter + i * size_channel + y * kernel_x + x];
        }
      }
    }
  }

  return filters_to_send;
}

// RSCK ->CKRS
float *Transform_Filters_Memory_b(const float *weights, const int K,
                                  const int G, const int C, const int R,
                                  const int S) {

  const int n_channels = C / G;
  const int kernel_y = R;
  const int kernel_x = S;
  const int n_filters = K; // this->num_output_;
  int size_channel = kernel_y * kernel_x;
  int size_filter = size_channel * n_channels;
  int n = size_filter * n_filters;

  float *filters_to_send =
      new float[n]; // Creating piece of memory that will use the simulator
  for (int n_f = 0; n_f < n_filters; n_f++) {
    for (int i = 0; i < n_channels; i++) {
      for (int y = 0; y < kernel_y; y++) {
        for (int x = 0; x < kernel_x; x++) {
          filters_to_send[n_f * size_filter + i * size_channel + y * kernel_x +
                          x] = weights[n_f * size_filter +
                                       (n_channels * (y * kernel_x + x)) + i];
        }
      }
    }
  }

  return filters_to_send;
}
