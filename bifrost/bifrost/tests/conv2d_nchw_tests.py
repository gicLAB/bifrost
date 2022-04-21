from unittest import TestCase

import bifrost
from bifrost.stonne.simulator import config_simulator

import tvm
from tvm import te, autotvm, relay
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm.relay import testing


class TestConv2d(TestCase):
    def setUp(self):
        out_channels = 2
        batch_size = 1

        # Let’s create a very simple network for demonstration.
        # It consists of convolution

        data = relay.var("data",
                         relay.TensorType((batch_size, 2, 10, 10), "float32"))
        weight = relay.var("weight")

        simple_net = relay.nn.conv2d(data=data,
                                     weight=weight,
                                     kernel_size=(3, 3),
                                     channels=out_channels,
                                     padding=(1, 1))
        data_shape = (batch_size, 2, 10, 10)
        net, params = testing.create_workload(simple_net)

        # Generate the data to resuse with both llvm and llvm stonne
        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

        # Run convolution on CPU to get output
        target = "llvm"
        lib = relay.build_module.build(net, target, params=params)
        ctx = tvm.context(target, 0)
        module = runtime.GraphModule(lib["default"](ctx))
        module.set_input("data", data)
        module.run()
        out_shape = (batch_size, out_channels, 10, 10)
        out = module.get_output(0, tvm.nd.empty(out_shape))
        self.out_llvm = out.asnumpy()

        # Build and run with llvm backend, but this time use the
        # stonne conv2d ops
        target = "llvm -libs=stonne"
        lib = relay.build_module.build(net, target, params=params)
        ctx = tvm.context(target, 0)
        self.module = runtime.GraphModule(lib["default"](ctx))
        self.module.set_input("data", data)

        self.out_shape = (batch_size, out_channels, 10, 10)

    def test_maeri(self):
        """
        Test running on MAERI
        """

        config_simulator(
            ms_size=16,
            reduce_network_type="ASNETWORK",
            ms_network_type="LINEAR",
            accumulation_buffer_enabled=False,
            dn_bw=8,
            rn_bw=8,
            controller_type="MAERI_DENSE_WORKLOAD",
        )

        self.module.run()
        out_stonne = self.module.get_output(0, tvm.nd.empty(
            self.out_shape)).asnumpy()

        # Check if output is equivalent to running the convolution on CPU
        self.assertTrue(
            np.all(np.round(out_stonne, 4) == np.round(self.out_llvm, 4)))

    def test_conv2d_tpu(self):

        config_simulator(
            ms_size=16,
            reduce_network_type="TEMPORALRN",
            ms_network_type="OS_MESH",
            accumulation_buffer_enabled=True,
            dn_bw=8,
            rn_bw=8,
            controller_type="TPU_OS_DENSE",
            sparsity_ratio=0,
        )
        self.module.run()
        out_stonne = self.module.get_output(0, tvm.nd.empty(
            self.out_shape)).asnumpy()

        # Check if output is equivalent to running the convolution on CPU
        self.assertTrue(
            np.all(np.round(out_stonne, 4) == np.round(self.out_llvm, 4)))

    def test_conv2d_sigma_sparse(self):
        config_simulator(
            ms_size=16,
            reduce_network_type="ASNETWORK",
            ms_network_type="LINEAR",
            accumulation_buffer_enabled=True,
            dn_bw=8,
            rn_bw=8,
            controller_type="SIGMA_SPARSE_GEMM",
            sparsity_ratio=20,
        )

        self.module.run()
        out_stonne = self.module.get_output(0, tvm.nd.empty(
            self.out_shape)).asnumpy()

        # Check if output is equivalent to running the convolution on CPU
        self.assertTrue(
            np.all(np.round(out_stonne, 4) == np.round(self.out_llvm, 4)))
