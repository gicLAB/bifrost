"""
Try to load STONNE as an external lib for conv2d

The smae way cuDNN or cuBLAS would be used with tvm.relay

"""

import tvm
from tvm import te, autotvm ,relay
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm.relay import testing
import logging
import random

# Import this add stonne as an x86 co-processor
import bifrost
from bifrost.stonne.simulator import architecture

import os

""" Example of using MAERI """
architecture.ms_size = 128
architecture.rn_bw = 64
architecture.dn_bw = 64
architecture.use_mrna = True
architecture.controller_type = "MAERI_DENSE_WORKLOAD"
architecture.create_config_file()

"""    Example of using TPU
architecture.ms_size = 128
architecture.rn_bw = 64
architecture.dn_bw = 64
architecture.controller_type = "SIGMA_SPARSE_GEMM"
architecture.sparsity_ratio = 0
architecture.create_config_file()
"""

"""    Example of using TPU
architecture.ms_cols = 128
architecture.ms_rows = 128
architecture.reduce_network_type = "TEMPORALRN"
architecture.ms_network_type = "OS_MESH"
architecture.accumulation_buffer_enabled = True
architecture.controller_type = "TPU_OS_DENSE"
architecture.create_config_file()
"""


out_channels = 2
batch_size = 1

# Let's create a very simple network for demonstration.
# It consists of convolution

data = relay.var("data", relay.TensorType((batch_size, 2, 50, 50), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(5, 5), channels=out_channels, padding=(1, 1),
)




data_shape = (batch_size, 2, 50, 50)
net, params = testing.create_workload(simple_net)

# Generate the data to resuse with both llvm and llvm stonne
np.random.seed(1)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# Build and run with llvm backend

target = "llvm"
lib = relay.build_module.build(net, target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 48, 48)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_llvm = out.asnumpy()

print(out_llvm)
# Build and run with llvm backend, but this time use the
# stonne conv2d ops

target = "llvm -libs=stonne"
lib = relay.build_module.build(net, target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 48, 48)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_stonne = out.asnumpy()

print(np.all(np.round(out_stonne, 4) == np.round(out_llvm, 4)))

print(np.round(out_stonne,4) == np.round(out_llvm, 4))

print(out_llvm.shape)
print(out_stonne.shape)
print(data.shape)

print(np.sum(out_llvm))
print(np.sum(out_stonne))

