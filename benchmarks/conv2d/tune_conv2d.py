from bifrost.stonne.simulator import config_simulator, architecture

config_simulator(
    ms_size=16,
    reduce_network_type="ASNETWORK",
    ms_network_type= "LINEAR",
    accumulation_buffer_enabled = False,
    dn_bw=8,
    rn_bw=8,
    controller_type="MAERI_DENSE_WORKLOAD",

)

architecture.tune = True
architecture.tuner.tune_convolutions_tile = True
architecture.tuner.conv_num = 30
architecture.tuner.tune_ms_size = True

if __name__ == "__main__":

    import tvm
    from tvm import te, autotvm ,relay, rpc
    import numpy as np
    from tvm.contrib import graph_runtime as runtime
    from tvm.relay import testing
    import logging
    import random

    np.random.seed(seed=1)


    # Import this add stonne as an x86 co-processor
    import bifrost
    from bifrost.tuner.stonne_builder import StonneLocalBuilder, StonneLocalRunner

    from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
    from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner


#
    architecture.tune = True

    out_channels = 2
    batch_size = 1

    # Let's create a very simple network for demonstration.
    # It consists of convolution, batch normalization, and ReLU activation.

    data = relay.var("data", relay.TensorType((batch_size, 2, 10, 10), "float32"))
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")

    simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(5,5), channels=out_channels, padding=(1, 1)
    )

    data_shape = (batch_size, 2, 10, 10)
    net, params = testing.create_workload(simple_net)

    # Generate the data to resuse with both llvm and llvm stonne
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    # Build and run with llvm backend
    logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion



    # Build and run with llvm backend, and use the
    # stonne conv2d ops
    target = "llvm --libs=stonne"

    mod, params = testing.create_workload(simple_net)

    log_file = "evaluation/conv2d.log"

    tuning_options = {
        "log_filename": log_file,
        "tuner": "random",
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
        tuner="gridsearch", 
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
