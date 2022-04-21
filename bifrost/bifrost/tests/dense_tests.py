from unittest import TestCase

import bifrost
from bifrost.stonne.simulator import config_simulator

import tvm
from tvm import te, autotvm ,relay
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm.relay import testing


class TestDense(TestCase):
    
    def setUp(self):
        # Let’s create a very simple network for demonstration.
        data = relay.var("data", shape=(1, 2))
        weight = relay.var("weight", shape=(6, 2))
    
        simple_net = relay.nn.dense(
            data=data, weight=weight
        )
    
        data_shape = (1,2)
        net, params = testing.create_workload(simple_net)
    
        # Generate the data to resuse with both llvm and llvm stonne
        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    
        target = "llvm"
        lib = relay.build_module.build(net, target, params=params)
    
        ctx = tvm.context(target, 0)
        module = runtime.GraphModule(lib["default"](ctx))
        module.set_input("data", data)
        module.run()
    
        self.out_shape = (1,6)
        out = module.get_output(0, tvm.nd.empty(self.out_shape))
        self.out_llvm = out.asnumpy()
        # Build and run with llvm backend, but this time use the
        #stonne conv2d ops
        
        target = "llvm -libs=stonne"
        lib = relay.build_module.build(net, target, params=params)
    
        ctx = tvm.context(target, 0)
        self.module = runtime.GraphModule(lib["default"](ctx))
        self.module.set_input("data", data)
        




    def test_maeri(self):
        """
        Test running on MAERI
        """
        
        config_simulator(
            ms_size=16,
            reduce_network_type="ASNETWORK",
            ms_network_type= "LINEAR",
            accumulation_buffer_enabled = False,
            dn_bw=8,
            rn_bw=8,
            controller_type="MAERI_DENSE_WORKLOAD",

        )

        self.module.run()
        out_stonne = self.module.get_output(
            0,
            tvm.nd.empty(self.out_shape)
            ).asnumpy()
        
        # Check if output is equivalent to running the convolution on CPU 
        self.assertTrue(np.all(np.round(out_stonne, 4) == np.round(self.out_llvm, 4)))

    def test_sigma_sparse(self):
        config_simulator(
            ms_size=16,
            reduce_network_type="ASNETWORK",
            ms_network_type= "LINEAR",
            accumulation_buffer_enabled = False,
            dn_bw=8,
            rn_bw=8,
            controller_type="SIGMA_SPARSE_GEMM",
            sparsity_ratio = 20,
        )

        self.module.run()
        out_stonne = self.module.get_output(
            0,
            tvm.nd.empty(self.out_shape)
            ).asnumpy()
        
        # Check if output is equivalent to running the convolution on CPU 
        self.assertTrue(np.all(np.round(out_stonne, 4) == np.round(self.out_llvm, 4)))
