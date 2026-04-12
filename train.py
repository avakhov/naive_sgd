import os
import random
import json
from nn import SimpleNN, points, sgd, gd

random.seed(123)

os.makedirs("out", exist_ok=True)

dataset = points("heart", 100)

n = SimpleNN(n0=1, n1=20, n2=20, n3=2)
sgd(n, dataset, lr=0.3, epochs=100)

target_x = [row[1] for row in dataset]
target_y = [row[2] for row in dataset]

output = {
    "target": {"x": target_x, "y": target_y},
    "snapshots": [
        {"epoch": epoch, "x": net_x, "y": net_y}
        for epoch, (net_x, net_y) in n.snapshots
    ],
}

with open("out/output.json", "w") as f:
    json.dump(output, f)

print("out/output.json saved.")
