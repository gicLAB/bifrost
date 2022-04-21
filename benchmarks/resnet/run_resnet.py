from os import pathsep
import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime

# Import this add stonne as an x86 co-processor
import bifrost
from bifrost.stonne.simulator import architecture
from bifrost.runner.run import run_torch
from vgg import vgg11_torch, input_batch

################################################################################
# Choose eval settings here
################################################################################

# choose maeri or sparse
architecture_setting = "sparse"

# chosoe sparsity ratio (ignored if not sigma)
sparsity_ratio = 0

#################################################################################
# Do not change anything after this
architecture.ms_size = 128
architecture.dn_bw=64
architecture.rn_bw=64
if architecture == "sparse":
    architecture.controller_type = "SIGMA_SPARSE_GEMM"
    architecture.sparsity_ratio = 0

architecture.create_config_file()

#architecture.load_tile_config(
#    conv_cfg_paths = conv_paths,
#    fc_cfg_paths = fc_paths
#    )

# Download an example image from the pytorch website

import time 
start = time.time()

run_torch(vgg11_torch, input_batch)

end = time.time()
print(end - start)