from os import pathsep
import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime

# Import this add stonne as an x86 co-processor
import bifrost
from bifrost.stonne.simulator import architecture
from bifrost.runner.run import run_torch
from alexnet import alex_model

################################################################################
# Choose eval settings here
################################################################################

# choose "maeri", "sparse", or "tpu"
architecture_setting = "sparse"

# chosoe sparsity ratio (ignored if not sigma)
sparsity_ratio = 50

# choose tile config: performance, opt, stonne_paper, basic
# If using sparse, these will be ignored.
tiles_conv = "performance"
tiles_fc = "performance"
architecture.ms_size = 128

#################################################################################
# Do not change anything after this
architecture.dn_bw=64
architecture.rn_bw=64
if architecture_setting == "sparse":
    architecture.controller_type = "SIGMA_SPARSE_GEMM"
    architecture.sparsity_ratio = 0
elif architecture_setting == "tpu":
    architecture.reduce_network_type = "TEMPORALRN"
    architecture.ms_network_type = "OS_MESH"
    architecture.accumulation_buffer_enabled = True
    architecture.controller_type = "TPU_OS_DENSE"
if sparsity_ratio:
    if sparsity_ratio == 50:
        from weight_pruning import model as alex_model
        architecture.sparsity_ratio = 50
    else:
        architecture.sparsity_ratio = 0


architecture.create_config_file()


conv_paths = [
    "tiles/%s/conv_1.txt" % tiles_conv,
    "tiles/%s/conv_2.txt" % tiles_conv,
    "tiles/%s/conv_3.txt" % tiles_conv,
    "tiles/%s/conv_4.txt" % tiles_conv,
    "tiles/%s/conv_5.txt" % tiles_conv
]
fc_paths = [
    "tiles/%s/fc_1.txt" % tiles_fc,
    "tiles/%s/fc_2.txt" % tiles_fc,
    "tiles/%s/fc_3.txt" % tiles_fc, 
]
architecture.load_mapping(
    conv = conv_paths,
    fc = fc_paths
    )

# Download an example image from the pytorch website
import urllib
from PIL import Image
from torchvision import transforms
import torch


url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

import time 
start = time.time()

run_torch(alex_model, input_batch)

end = time.time()
print(end - start)