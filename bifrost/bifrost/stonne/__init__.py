import tvm

from .connect_stonne import load_lib
from .tiles import conv_tiles, fc_tiles

# Register STONNE library
_LIB = load_lib()

# Register the ops
from . import ops

