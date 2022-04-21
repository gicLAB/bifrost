import json

ms_size_64_cycles  = list()
ms_size_64_psums = list()
path = "alexnet.log"

best = {}
tr = {}

with open(path, "r") as f:
    for line in f:
        config = json.loads(line)
        
        layer = tuple(config["input"][2][1][1])
        result = config["result"][2]
        config = config["config"]["entity"]
        if result < 0:
            continue
        
        if layer in best:
            if best[layer] > result:
                best[layer] = result
                tr[layer] = config
        else:
            best[layer] = result
            tr[layer] = config

for k,v in tr.items():
    print(k,v)

for k,v in best.items():
    print(k,v)