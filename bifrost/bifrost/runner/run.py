import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime

import torch
from torchvision import transforms


def run_torch(torch_model, input):
    torch_model.eval()
    trace = torch.jit.trace(torch_model, input).eval()
    mod, params = relay.frontend.from_pytorch(trace, [("trace", input.shape)])
    target = "llvm -libs=stonne"
    lib = relay.build(mod, target=target, params=params)
    ctx = tvm.context(target, 0)
    module = runtime.GraphModule(lib["default"](ctx))
    module.set_input("trace", input)
    module.run()
    return module.get_output(0)


def run_onnx(onnx_model, input):
    shape_dict = {"1": input.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    target = "llvm -libs=stonne"
    lib = relay.build(mod, target=target, params=params)
    ctx = tvm.context(target, 0)
    module = runtime.GraphModule(lib["default"](ctx))
    module.set_input("trace", input)
    module.run()
    return module.get_output(0)
