import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("output.json") as f:
    data = json.load(f)

target = data["target"]
snapshots = data["snapshots"]

plt.figure(figsize=(6.5, 6.5), dpi=100)
plt.plot(target["x"], target["y"], color='red', label='target')

num_snapshots = len(snapshots)
for i, snap in enumerate(snapshots):
    is_last = i == num_snapshots - 1
    alpha = 0.1 + 0.9 * (i / (num_snapshots - 1)) if num_snapshots > 1 else 1.0
    color = 'green' if is_last else 'blue'
    label = f'epoch {snap["epoch"]}' if is_last else None
    lw = 3.0 if is_last else 1.0
    plt.plot(snap["x"], snap["y"], color=color, alpha=alpha, label=label, linewidth=lw)

plt.axis('equal')
plt.legend()
plt.grid()
file = 'graph.png'
plt.savefig(file)
print(file + " saved.")
