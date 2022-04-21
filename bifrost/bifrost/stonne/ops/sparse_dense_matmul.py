""" 
Register everything to do with dense
"""
import tvm
from tvm import te, relay, autotvm
from tvm.topi import generic
import tvm.relay.op as _op
from tvm.relay.op.strategy.generic import *
import os
from ..simulator import architecture
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.te import SpecializedCondition


@autotvm.register_topi_compute("sparse_dense_stonne.x86")
def sparse_dense_stonne(cfg, data, weight, sparse_lhs=False):

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


@autotvm.register_topi_schedule("sparse_dense_stonne.x86")
def schedule_sparse_dense_stonne(cfg, outs):
    return te.create_schedule([x.op for x in outs])


@override_native_generic_func("sparse_dense_strategy")
def sparse_dense_strategy(attrs, inputs, out_type, target):
    """sparse dense generic strategy"""
    logger.warning("sparse dense is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_dense(topi.nn.sparse_dense),
        wrap_topi_schedule(topi.generic.schedule_sparse_dense),
        name="sparse_dense.generic",
    )

    if "stonne" in target.libs:
        strategy.add_implementation(
            wrap_compute_sparse_dense(sparse_dense_stonne),
            wrap_topi_schedule(schedule_sparse_dense_stonne),
            name="dense_stonne.x86",
            plevel=12,
        )
    return strategy
