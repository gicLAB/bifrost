import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import argparse
from tqdm import tqdm


def remove_weight_prune_masks(model, exclude_list=[]):
    for name, m in model.named_modules():
        m.auto_name = name
        if type(m) in [nn.modules.conv.Conv2d, nn.modules.Linear]:
            if name not in exclude_list:
                prune.remove(m, "weight")


def weight_prune(model, prune_rate=0.5):
    # build up list on conv2d layer names, and give
    # each model layer a name we can look up later
    exclude_list = []
    prune_names = []

    for name, m in model.named_modules():
        m.auto_name = name
        if type(m) in [nn.modules.conv.Conv2d, nn.modules.Linear]:
            if name not in exclude_list:
                prune_names.append(name)

    # create a tuple of layers we will prune
    parameters_to_prune = []
    for m in model.modules():
        if m.auto_name in prune_names:
            parameters_to_prune.append((m, "weight"))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_rate,
    )

    return model


def verify_pruning(model):
    zero_params = 0.0
    all_params = 0.0
    for name, m in model.named_modules():
        if type(m) in [nn.modules.conv.Conv2d, nn.modules.Linear]:
            zero_params += torch.sum(m.weight == 0)
            all_params += m.weight.nelement()
    return 100 * float(zero_params) / float(all_params)


def main(args):
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    model = torch.hub.load("pytorch/vision:v0.6.0", "alexnet", pretrained=True)
    model.eval()

    prune_rates = [
        50,
    ]
    for prune_rate in tqdm(prune_rates):
        # add masks to remove smallest weights
        model = weight_prune(model, prune_rate / 100)

        # remove masked weights
        remove_weight_prune_masks(model)

        ## save model
        # fname = f"{args.save_directory}/{args.save_file}_{prune_rate}.pt"
        # torch.save(model, fname)

        measured_sparsity = verify_pruning(model)
        print(f"Actual sparsity of {prune_rate} model is {measured_sparsity}")
        # model.load_state_dict(torch.load(fname)) <-- be sure to load the sparse weights in TVM,
        # you may have an error I couldn't solve:
        # torch.nn.modules.module.ModuleAttributeError: 'AlexNet' object has no attribute 'copy'
        # in which case, copy the code from this script and use it with your own code directly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Pruning")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument(
        "--save_file", default="alexnet", type=str, help="save file for checkpoints"
    )
    parser.add_argument(
        "--save_directory",
        default="models/",
        type=str,
        help="save directory for checkpoints",
    )

    args = parser.parse_args()

    main(args)
