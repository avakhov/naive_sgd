import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nn import SimpleNN, points, sgd, gd, adam, sgd_momentum

random.seed(123)

dataset = points("heart", 200)
t_list = [row[0] for row in dataset]
x = [row[1] for row in dataset]
y = [row[2] for row in dataset]

n = SimpleNN(n0=1, n1=20, n2=20, n3=2)
sgd_momentum(n, dataset, lr=0.1, epochs=300, snapshot_every=30)

plt.figure(figsize=(6.5, 6.5), dpi=100)
plt.plot(x, y, color='red', label='target')

num_snapshots = len(n.snapshots)
for i, (epoch, (net_x, net_y)) in enumerate(n.snapshots):
    is_last = i == num_snapshots - 1
    alpha = 0.1 + 0.9 * (i / (num_snapshots - 1)) if num_snapshots > 1 else 1.0
    color = 'green' if is_last else 'blue'
    label = f'epoch {epoch}' if is_last else None
    plt.plot(net_x, net_y, color=color, alpha=alpha, label=label)

plt.axis('equal')
plt.legend()
plt.grid()
file = 'graph.png'
plt.savefig(file)
print(file + " saved.")
