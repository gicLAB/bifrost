#include "Config.h"
#include "STONNEModel.h"
#include "include/Analyzer.h"
#include "include/DNNModel.h"
#include "include/MAERIModel.h"

void mRNA(Stonne *stonne_instance, Config stonne_cfg, Layer_t type, int X,
          int Y, int C, int R, int S, int X_, int Y_, int K, int N,
          int stride) {
  std::cout << "Create MAERI" << std::endl;
  mrna::Maeri *maeri = new mrna::Maeri(stonne_cfg.m_MSNetworkCfg.ms_size,
                                       stonne_cfg.m_SDMemoryCfg.n_read_ports,
                                       stonne_cfg.m_SDMemoryCfg.n_write_ports);

  std::cout << "Create DNN" << std::endl;
  mrna::DNNModel *dnn = new mrna::DNNModel();
  dnn->cnn_input->input_x = X;
  dnn->cnn_input->input_y = Y;
  dnn->cnn_input->input_channel = C;
  dnn->cnn_input->input_batch = 32;

  dnn->cnn_filter->filter_x = R;
  dnn->cnn_filter->filter_y = S;
  dnn->cnn_filter->filter_channel = C;
  dnn->cnn_filter->filter_number = K;
  dnn->cnn_filter->window_stride = stride;

  dnn->cnn_output->output_x = X_;
  dnn->cnn_output->output_y = Y_;
  dnn->cnn_output->output_channel = K;
  dnn->cnn_output->output_batch = 32;

  dnn->dnn_hidden->hidden_x = 0;
  dnn->dnn_hidden->hidden_y = 0;
  dnn->dnn_hidden->hidden_channel = 0;

  dnn->model_name = "";
  switch (type) {
  case CONV:
    dnn->layer_type = "CONV";
  case FC:
    dnn->layer_type = "FC";
  case GEMM:;
  case POOL:;
  }
  dnn->layer_num = "0";

  std::cout << "Create analyzer" << std::endl;
  mrna::Analyzer *analyzer = new mrna::Analyzer(maeri, dnn, mrna::performance);

  analyzer->setshowenergy(false);
  analyzer->setoptgoal(mrna::performance);

  std::string outputfile = "test.txt";
  std::ofstream Profile_result(outputfile);

  mrna::OptGoal opt_goal = mrna::performance;

  mrna::MappingStrategy *bestmap;
  std::cout << "Analyse" << std::endl;
  if (type == CONV) {
    analyzer->AnalyzeCNN(Profile_result, opt_goal);

    unsigned int T_R = analyzer->bestmap->kernel_x;
    unsigned int T_S = analyzer->bestmap->kernel_y;
    unsigned int T_C = analyzer->bestmap->kernel_c;
    unsigned int T_K = analyzer->bestmap->kernel_n;
    unsigned int T_G = 1;
    unsigned int T_N = analyzer->bestmap->kernel_in;
    unsigned int T_X_ = analyzer->bestmap->kernel_ox;
    unsigned int T_Y_ = analyzer->bestmap->kernel_oy;
    std::cout << "Load tile" << std::endl;
    stonne_instance->loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_);
  } else if (type == FC) {
    analyzer->AnalyzeFC(Profile_result, opt_goal);
  }
}
