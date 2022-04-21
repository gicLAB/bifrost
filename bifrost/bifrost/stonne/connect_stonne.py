import ctypes
import os


def load_lib():
    """
    Register the stonne connection library into TVM
    """

    # Find library based on relative paths
    stonne_conv2d = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "stonne_lib/stonne_lib.so")

    # load in as global so the global extern symbol is visible to other dll.
    lib = ctypes.CDLL(stonne_conv2d, ctypes.RTLD_GLOBAL)
    return lib
