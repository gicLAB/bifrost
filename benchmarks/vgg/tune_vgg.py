from bifrost.stonne.simulator import config_simulator, architecture


architecture.ms_size = 128
architecture.rn_bw = 64
architecture.dn_bw = 64
architecture.tune = True
architecture.tuner.tune_psums = True
architecture.tuner.conv_num = 20
architecture.tuner.tune_convolutions_tile = True
architecture.create_config_file()

if __name__ == "__main__":

    import tvm
    from tvm import te, autotvm ,relay, rpc
    import numpy as np
    from tvm.contrib import graph_runtime as runtime
    from tvm.relay import testing
    import logging
    import random
    import torch

    # Import this add stonne as an x86 co-processor
    import bifrost
    from bifrost.tuner.stonne_builder import StonneLocalBuilder, StonneLocalRunner

    from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
    from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
    from vgg import vgg11_torch as torch_model
    from vgg import input_batch

    torch_model.eval()
    trace = torch.jit.trace(torch_model, input_batch).eval()
    
    mod, params = relay.frontend.from_pytorch(
        trace, [("trace", input_batch.shape)])

    # Build and run with llvm backend, and use the
    # stonne conv2d ops
    target = "llvm --libs=stonne"
    log_file = "bifrost_temp/vgg.log"

    tuning_options = {
        "log_filename": log_file,
        "tuner": "xgb",
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=StonneLocalBuilder(),
            runner=StonneLocalRunner(
                number=0,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=True
            ),
        ),
    }
    batch_size = 1
    graph_opt_sch_file = "graph_opt.log" 
    input_name = "data"

    # You can skip the implementation of this function for this tutorial.
    def tune_kernels(
        tasks, 
        measure_option, 
        tuner="xgb", 
        early_stopping=None, 
        log_filename=log_file
    ):

        for i, task in enumerate(tasks):
            print(task)
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # create tuner
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(task, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(task, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(task)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(task)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            # do tuning
            n_trial = len(task.config_space)

            print(task.config_space)
            print(n_trial, "test")
            tuner_obj.tune(
                n_trial=n_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(100, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename),
                ],
            )




    remote = rpc.LocalSession()

    tasks = autotvm.task.extract_from_program(
            mod, 
            target=target, 
            params=params, 
            ops=(relay.op.get("nn.conv2d"),)
    )
    print(tasks)
    tune_kernels(tasks, tuning_options["measure_option"])