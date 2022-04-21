import json

path = 'conv2d.log'

best = {}
worst = {}
tr = {}

with open(path, "r") as f:
    for line in f:
        config = json.loads(line)
        
        ms_size = tuple(config["config"]["entity"][8])
        result = config["result"][2]
        config = config["config"]["entity"]
        if result < 0:
            continue
        
        if ms_size in best:
            if best[ms_size] > result:
                best[ms_size] = result
                #tr[ms_size] = config
        else:
            best[ms_size] = result
            tr[ms_size] = config

        if ms_size in worst:
            if worst[ms_size] < result and result !=100000000000:
                worst[ms_size] = result
                #tr[ms_size] = config
        elif result != 100000000000:
            worst[ms_size] = result
            tr[ms_size] = config
        

        

for k,v in worst.items():
    print(k,v)

for k,v in best.items():
    print(k,v)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.6  # previous pdf hatch linewidth
mpl.rcParams['hatch.linewidth'] = 6.0  # previous svg hatch linewidth
 
nice_fonts = {
    "text.usetex": False,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 30,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 30,
    "xtick.labelsize": 20,
    "ytick.labelsize": 30,
}
 
mpl.rcParams.update(nice_fonts)





labels = [8,16,32,64,128]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
print(list(worst.values()))
print(list(best.values()))
# Create a figure, set aspect ratio to golden ratio
plt.figure(figsize=((1+(5**1/2))/2*10,10))
plt.bar(x - width/2, worst.values(), width, label='Suboptimal Mapping')
plt.bar(x + width/2, best.values(), width, label='Optimal Mapping')
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xticks(x, labels)
plt.ylabel('Clock Cycles')
plt.xlabel('The number of mulitpliers')
plt.yscale('log')
plt.legend()
plt.savefig("conv2d_maeri.pdf", bbox_inches='tight')

plt.show()