import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nn import SimpleNN, points, sgd, gd

random.seed(123)

dataset = points("heart", 200)
t_list = [row[0] for row in dataset]
x = [row[1] for row in dataset]
y = [row[2] for row in dataset]

n = SimpleNN(n0=1, n1=20, n2=20, n3=2)
sgd(n, dataset, lr=0.3, epochs=500, snapshot_every=10)

# graph
plt.figure(figsize=(6.5, 6.5), dpi=100)
plt.plot(x, y, color='red', label='target')

num_snapshots = len(n.snapshots)
for i, (epoch, (net_x, net_y)) in enumerate(n.snapshots):
    alpha = 0.1 + 0.9 * (i / (num_snapshots - 1)) if num_snapshots > 1 else 1.0
    label = f'epoch {epoch}' if i == num_snapshots - 1 else None
    plt.plot(net_x, net_y, color='blue', alpha=alpha, label=label)

plt.axis('equal')
plt.legend()
plt.grid()
file = 'graph.png'
plt.savefig(file)
print(file + " saved.")
