""" 
Register everything to do with matmul
"""
import tvm
from tvm import te, relay, autotvm
from tvm.topi import generic
import tvm.relay.op as _op
from tvm.relay.op.strategy.generic import *
import os
from ..simulator import architecture
from ..tiles import tiles
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.te import SpecializedCondition


@autotvm.register_topi_compute("matmul_stonne.x86")
def matmul_stonne(cfg, x, y, out_shape=None):
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    return te.extern(
        (M, N),
        [data, weight],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.stonne.dense",
            architecture.path,  # [0] Architecture file
            M,  # [1] Batch size
            K,  # [2] Number of input neurons
            N,  # [3] Number of output neurons
            tiles.path,  # [4] Tiles path
            architecture.sparsity_ratio,  # [5]
            architecture.print_stats,  # [6] Create stats output files
            ins[0],  # [7] Data
            ins[1],  # [8] Weight
            outs[0],  # [9] Output
        ),
        name="d",
        dtype=out_dtype)


@autotvm.register_topi_schedule("matmul_stonne.x86")
def schedule_matmul_stonne(cfg, outs):
    return te.create_schedule([x.op for x in outs])


@batch_matmul_strategy.register("cpu")
def batch_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """batch_matmul x86 strategy"""
    strategy = _op.OpStrategy()
    if is_dynamic(out_type) or is_auto_scheduler_enabled():
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.nn.batch_matmul,
                                      need_auto_scheduler_layout=True),
            wrap_topi_schedule(topi.generic.nn.schedule_batch_matmul),
            name="batch_matmul.generic",
            plevel=10,
        )
    else:
        strategy.add_implementation(
            wrap_compute_batch_matmul(matmul_stonne),
            wrap_topi_schedule(schedule_matmul_stonne),
            name="batch_matmul.x86",
            plevel=10,
        )

    if "stonne" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(dense_stonne),
            wrap_topi_schedule(schedule_dense_stonne),
            name="dense_stonne.x86",
            plevel=15,
        )

    if "cblas" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.x86.batch_matmul_cblas),
            wrap_topi_schedule(topi.x86.schedule_batch_matmul_cblas),
            name="batch_matmul_cblas.x86",
            plevel=15,
        )
    if "mkl" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.x86.batch_matmul_mkl),
            wrap_topi_schedule(topi.x86.schedule_batch_matmul_mkl),
            name="batch_matmul_mkl.x86",
            plevel=15,
        )
    return strategy
