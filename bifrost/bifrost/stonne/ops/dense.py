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
import random
from .. import fc_tiles


@autotvm.register_topi_compute("dense_stonne.x86")
def dense_stonne(cfg, data, weight, units=None, out_dtype=""):
    """Dense operator
    Applies a linear transformation

    .. math::

    `Y = X * W^T`

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator,
        of shape `(d_1, d_2, ..., d_n, units_in)`.

    weight : tvm.relay.Expr
        The weight expressions, 2-D matrix,
        of shape `(units, units_in)`.

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense,
        of shape `(d_1, d_2, ..., d_n, units)`.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    # Get the name to store the costs
    dirname = os.path.dirname(__file__)
    costs_path = os.path.join(dirname, "../data/costs.json")

    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)

    # Define tuning space
    if architecture.tune:

        # Generate the different fc tile options
        if architecture.tuner.tune_fc_tile:
            architecture.tuner.fc_tile()

        # Get and register the tuning knobs
        knobs = architecture.tuner.create_knobs(dense=True)
        for knob in knobs:
            cfg.define_knob(*knob)

        # Create the architecture files
        architecture.config(cfg, dense=True)

    # Choose tiles
    elif architecture.manual_tile_paths:
        architecture.set_manual_tile_config("FC")

    if not architecture.manual_tile_paths and not architecture.tuner.tune_fc_tile:
        architecture.fc_tiles_path = architecture.fc_tiles.generate_basic_tile_config(
        )

    return te.extern(
        (M, N),
        [data, weight],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.stonne.dense",
            architecture.path,  # [0] Architecture file
            M,  # [1] 
            K,  # [2] 
            N,  # [3] 
            architecture.fc_tiles_path,  # [4] Tiles path
            architecture.sparsity_ratio,  # [5]
            architecture.print_stats,  # [6] Create stats output files
            architecture.tune,  # [7] Enable if tuning
            architecture.tuner.tune_psums,  # [8] 
            "fc_" + str(random.randrange(10000000)),  # [9]
            costs_path,  # [10]
            ins[0],  # [11] Data
            ins[1],  # [12] Weight
            outs[0],  # [13] Output
        ),
        name="a",
        dtype=out_dtype)


@autotvm.register_topi_schedule("dense_stonne.x86")
def schedule_dense_stonne(cfg, outs):
    cfg.add_flop(1)
    return te.create_schedule([x.op for x in outs])


@dense_strategy.register("cpu")
def dense_strategy_cpu(attrs, inputs, out_type, target):
    """dense x86 strategy"""
    strategy = _op.OpStrategy()
    m, _ = inputs[0].shape
    same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
    dtype = inputs[0].dtype
    u8s8s32 = dtype == "uint8" and inputs[
        1].dtype == "int8" and out_type.dtype == "int32"

    if "stonne" in target.libs:
        strategy.add_implementation(
            wrap_compute_dense(dense_stonne),
            wrap_topi_schedule(schedule_dense_stonne),
            name="dense_stonne.x86",
        )
    else:
        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_nopack),
            wrap_topi_schedule(topi.x86.schedule_dense_nopack),
            name="dense_nopack.x86",
        )
    return strategy
