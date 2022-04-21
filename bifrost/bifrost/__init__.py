from . import stonne
import os
import json

# Create a place to store output files
if not os.path.exists("bifrost_temp"):
    os.mkdir("bifrost_temp")
    cycles = {"layer": [], "value": []}

    # Use this to store cycle values
    json_path = os.path.join(os.getcwd(), "bifrost_temp/cycles.json")
    with open(json_path, "w+") as f:
        json.dump(cycles, f)
