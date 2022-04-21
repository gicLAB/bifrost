import shutil
import time 
import tempfile
import os
import json
from random import getrandbits
from collections import namedtuple
import numpy as np

from tvm import nd
from tvm.error import TVMError
from tvm.autotvm.measure.measure import MeasureResult, MeasureErrorNo, Builder
from tvm.autotvm.measure.measure_methods import _build_func_common, request_remote, logger, Runner, check_remote
from tvm.autotvm.utils import get_const_tuple
from tvm.autotvm.measure.local_executor import LocalExecutor
from tvm.autotvm.task.space import InstantiationError
from tvm.contrib import tar
from tvm.target import Target

from ..stonne.simulator import architecture

class StonneLocalBuilder(Builder):
    """Run compilation on local machine

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    build_func: callable or str
        If is 'default', use default build function
        If is 'ndk', use function for android ndk
        If is callable, use it as custom build function, expect lib_format field.
    """

    def __init__(self, timeout=10, n_parallel=None, build_func="default"):
        super(StonneLocalBuilder, self).__init__(timeout, n_parallel)

        if isinstance(build_func, str):
            build_func = tar.tar

        self.build_func = _StonneWrappedBuildFunc(build_func)
        self.executor = LocalExecutor(timeout=timeout)
        self.tmp_dir = tempfile.mkdtemp()

    def build(self, measure_inputs):
        results = []

        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i : i + self.n_parallel]:
                ret = self.executor.submit(self.build_func, inp, self.tmp_dir, **self.build_kwargs)
                futures.append(ret)

            for future in futures:
                res = future.get()

                if isinstance(res, Exception):
                    # timeout or fleet error, return MeasureResult directly
                    results.append(
                        MeasureResult(
                            (res,), MeasureErrorNo.BUILD_TIMEOUT, self.timeout, time.time()
                        )
                    )
                elif res.error is not None:
                    # instantiation error
                    if isinstance(res.error, InstantiationError):
                        results.append(
                            MeasureResult(
                                (res.error,),
                                MeasureErrorNo.INSTANTIATION_ERROR,
                                res.time_cost,
                                time.time(),
                            )
                        )
                    else:
                        if "InstantiationError" in str(res.error):
                            msg = str(res.error)
                            try:
                                msg = msg.split("\n")[-2].split(": ")[1]
                            except Exception:  # pylint: disable=broad-except
                                pass
                            results.append(
                                MeasureResult(
                                    (InstantiationError(msg),),
                                    MeasureErrorNo.INSTANTIATION_ERROR,
                                    res.time_cost,
                                    time.time(),
                                )
                            )
                        else:  # tvm error
                            results.append(
                                MeasureResult(
                                    (res.error,),
                                    MeasureErrorNo.COMPILE_HOST,
                                    res.time_cost,
                                    time.time(),
                                )
                            )
                else:
                    # return BuildResult
                    results.append(res)

        return results


class _StonneWrappedBuildFunc:
    """
    Wrap build_func to a function that can be used in measure.

    Note: this is a class instead of a closure so that it can be pickled when
    using multiprocessing.

    Parameters
    ----------
    build_func : The compilation function
        We expect fcompile to contain an attr "output_format"

    Returns
    -------
    wrapped_build_func : callable
        The wrapped build function
    """

    def __init__(self, build_func):
        if not hasattr(build_func, "output_format"):
            raise AttributeError("Expect build_func to have the attribute output_format.")
        self.build_func = build_func

    def __call__(self, measure_input, tmp_dir, **kwargs):
        """
        Wrapped build func.

        Parameters
        ----------
        measure_input: MeasureInput
            The input of measurement

        tmp_dir: str
            The path of temporary directory to export generated library
        """
        tic = time.time()
        try:
            # Find library based on relative paths
            # TODO: Modify this so that several fucntions can be uploaded
            dirname = os.path.dirname(__file__)
            filename_module = os.path.join(dirname, "../stonne/stonne_lib/stonne_lib.so")

            filename = os.path.join(
                tmp_dir, "tmp_func_%0x.%s" % (getrandbits(64), self.build_func.output_format)
            )
            func, arg_info = _build_func_common(measure_input, **kwargs)
            func.export_library(filename, self.build_func)
        except Exception as e:  # pylint: disable=broad-except
            return StonneBuildResult(None, None, e, time.time() - tic)
        return StonneBuildResult(filename, filename_module, arg_info, None, time.time() - tic)


class StonneBuildResult(namedtuple("BuildResult", ("filename", "filename_module", "arg_info", "error", "time_cost"))):
    """
    Stores all the necessary inputs for a measurement.

    Parameters
    ----------
    filename : str
        The filename of generated library
    arg_info : Tuple
        The shape and dtype information of tvm tensor arguments
    error : Exception
        The error happens during compilation.
    time_cost : float
        The time cost of building
    """

def run_stonne_through_rpc(
    measure_input,
    build_result,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    remote_args,
    ref_input=None,
    ref_output=None,
    enable_cpu_cache_flush=False,
):
    """Run a generated library through rpc

    Parameters
    ----------
    measure_input: MeasureInput
        The raw measure input
    build_result: BuildResult
        The result returned from Builder. This contains the path to the generated library.
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float
        The cool down interval between two measurements
    remote_args: Tuple
        The argument for request_remote
    ref_input: List of np.ndarray
        The reference input used for checking correctness
    ref_output: List of np.ndarray
        The reference output used for checking correctness
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    """
    if isinstance(build_result, MeasureResult):
        return build_result

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR
    try:
        # upload built module
        remote = request_remote(*remote_args)
        remote.upload(build_result.filename)
        remote.load_module(build_result.filename_module)
        func = remote.load_module(os.path.split(build_result.filename)[1])
        ctx = remote.context(str(measure_input.target), 0)

        # Limitation:
        # We can not get PackFunction directly in the remote mode as it is wrapped
        # under the std::function. We could lift the restriction later once we fold
        # the PackedFunc as an object. Currently, we pass function name to work
        # around it.
        f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
        time_f = func.time_evaluator(
            func.entry_name,
            ctx,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            f_preproc=f_prepare,
        )

        # set input
        if ref_input:
            args = [nd.array(x, ctx=ctx) for x in ref_input]
        else:
            try:
                random_fill = remote.get_function("tvm.contrib.random.random_fill")
            except AttributeError:
                raise AttributeError(
                    "Please make sure USE_RANDOM is ON in the config.cmake " "on the remote devices"
                )
            args = [nd.empty(x[0], dtype=x[1], ctx=ctx) for x in build_result.arg_info]
            for arg in args:
                random_fill(arg)
            ctx.sync()

        # Run this to create the output files
        time_f(*args).results
        
        # Get the costs from stonne
        dirname = os.path.dirname(__file__)
        cost_file = os.path.join(dirname, "../stonne/data/costs.json")
        with open(cost_file, "r+") as f:
            json_dict = json.load(f)
            costs = json_dict["value"]
            # Reset JSON file
            f.seek(0)
            f.truncate()
            json_dict["value"] = []
            json_dict["tuning_name"] = "null"
            json.dump(json_dict, f)


        # clean up remote files
        remote.remove(build_result.filename)
        remote.remove(os.path.splitext(build_result.filename)[0] + ".so")
        remote.remove("")

        # check correctness of output
        if ref_output:
            for expected, real in zip(ref_output, args):
                if not np.allclose(expected, real.asnumpy(), rtol=1e-4):
                    logger.warning("Wrong Answer!")
                    errno = MeasureErrorNo.WRONG_ANSWER
        average = costs[0]
    except TVMError as exc:
        msg = str(exc)
        if "Stack trace returned" in msg:
            msg = msg[: msg.index("Stack trace returned")]
        if "CUDA Source" in msg:
            msg = msg[: msg.index("CUDA Source")]
        average = 100000000000
        costs = (RuntimeError(msg[:1024]),)

        errno = MeasureErrorNo.RUNTIME_DEVICE
    tstamp = time.time()
    time.sleep(cooldown_interval)

    return MeasureResult(costs, errno, average, tstamp)


class StonneRPCRunner(Runner):
    """Run generated code on remove devices.
    This function will ask a RPC Tracker to get device for measurement.

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    key: str
        The key of the device registered in the tracker
    host: str
        The host address of RPC Tracker
    port: int
        The port of RPC Tracker
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    check_correctness: bool, optional
        Whether check correctness after measurement. This will use llvm cpu target to
        call your template and get the reference output.
        This can work for TOPI templates, but may not work for your custom template.
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    """

    def __init__(
        self,
        key,
        host,
        port,
        priority=1,
        timeout=2**16,
        n_parallel=None,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        check_correctness=False,
        enable_cpu_cache_flush=False,
    ):
        super(StonneRPCRunner, self).__init__(timeout, n_parallel)

        self.key = key
        self.host = host
        self.port = port
        self.priority = priority
        self.timeout = timeout

        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms

        self.ref_input = None
        self.ref_output = None
        self.enable_cpu_cache_flush = enable_cpu_cache_flush
        self.check_correctness = check_correctness
        self.cooldown_interval = cooldown_interval

        self.executor = LocalExecutor()

    def set_task(self, task):
        self.task = task

        if check_remote(task.target, self.key, self.host, self.port):
            logger.info("Get devices for measurement successfully!")
        else:
            raise RuntimeError(
                "Cannot get remote devices from the tracker. "
                "Please check the status of tracker by "
                "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                "and make sure you have free devices on the queue status."
            )

        if self.check_correctness:
            # use llvm cpu to generate a reference input/output
            # this option works for tuning topi, but might not work for you custom op
            with Target("llvm"):
                s, arg_bufs = task.instantiate(task.config_space.get(0))
            self.ref_input = [
                np.random.uniform(size=get_const_tuple(x.shape)).astype(x.dtype) for x in arg_bufs
            ]
            func = build(s, arg_bufs, "llvm")
            tvm_buf = [nd.array(x) for x in self.ref_input]
            func(*tvm_buf)
            self.ref_output = [x.asnumpy() for x in tvm_buf]

    def get_build_kwargs(self):
        kwargs = {}
        if (
            "cuda" in self.task.target.keys
            or "opencl" in self.task.target.keys
            or "rocm" in self.task.target.keys
            or "vulkan" in self.task.target.keys
        ):
            remote = request_remote(self.key, self.host, self.port)
            ctx = remote.context(str(self.task.target), 0)
            max_dims = ctx.max_thread_dimensions
            kwargs["check_gpu"] = {
                "max_shared_memory_per_block": ctx.max_shared_memory_per_block,
                "max_threads_per_block": ctx.max_threads_per_block,
                "max_thread_x": max_dims[0],
                "max_thread_y": max_dims[1],
                "max_thread_z": max_dims[2],
            }

            if "cuda" in self.task.target.keys:
                kwargs["cuda_arch"] = "sm_" + "".join(ctx.compute_version.split("."))
        if self.task.target.device_name == "micro_dev":
            kwargs.setdefault("build_option", {})["tir.disable_vectorize"] = True

        return kwargs

    def run(self, measure_inputs, build_results):
        results = []
        remote_args = (self.key, self.host, self.port, self.priority, self.timeout)

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(
                measure_inputs[i : i + self.n_parallel], build_results[i : i + self.n_parallel]
            ):
                ret = self.executor.submit(
                    run_stonne_through_rpc,
                    measure_inp,
                    build_res,
                    self.number,
                    self.repeat,
                    self.min_repeat_ms,
                    self.cooldown_interval,
                    remote_args,
                    self.ref_input,
                    self.ref_output,
                    self.enable_cpu_cache_flush,
                )
                futures.append(ret)
            for future in futures:
                res = future.get()
                if isinstance(res, Exception):  # executor error or timeout
                    results.append(
                        MeasureResult(
                            (str(res),), MeasureErrorNo.RUN_TIMEOUT, self.timeout, time.time()
                        )
                    )
                else:
                    results.append(res)

        return results


class StonneLocalRunner(StonneRPCRunner):
    """Run generated code on local devices.

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    check_correctness: bool, optional
        Whether check correctness after measurement. This will use llvm cpu target to
        call your template and get the reference output.
        This can work for TOPI templates, but may not work for your custom template.
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    Note
    ----
    This is a "fake" local mode. We start a silent rpc tracker and rpc server
    for the user. In this way we reuse timeout/isolation mechanism in RPC infrastructure.
    """

    def __init__(
        self,
        timeout=2**16,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        check_correctness=False,
        enable_cpu_cache_flush=False,
    ):
        super(StonneLocalRunner, self).__init__(
            "",
            None,
            None,
            0,
            timeout=timeout,
            n_parallel=1,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            cooldown_interval=cooldown_interval,
            check_correctness=check_correctness,
            enable_cpu_cache_flush=enable_cpu_cache_flush,
        )
        self.tracker = None
        self.server = None

    def set_task(self, task):
        # pylint: disable=import-outside-toplevel
        from tvm.rpc.tracker import Tracker
        from tvm.rpc.server import Server

        self.task = task
        tracker = Tracker("0.0.0.0", port=9000, port_end=10000, silent=True)
        device_key = "$local$device$%d" % tracker.port
        server = Server(
            "0.0.0.0",
            port=9000,
            port_end=10000,
            key=device_key,
            use_popen=True,
            silent=True,
            tracker_addr=(tracker.host, tracker.port),
        )
        self.key = device_key
        self.host = tracker.host
        self.port = tracker.port

        super(StonneLocalRunner, self).set_task(task)
        return server, tracker