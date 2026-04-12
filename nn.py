import random
import math
from autograd import Value

def points(fig, n):
    ta = [t / n for t in range(n + 1)]
    out = []
    if fig == "heart":
        for t in ta:
            angle = t * 2 * math.pi
            x = 0.5 * math.sin(angle) ** 3
            y = (13 * math.cos(angle) - 5 * math.cos(2 * angle) - 2 * math.cos(3 * angle) - math.cos(4 * angle)) / 30
            out.append([t, x, y])
    else:
        raise ValueError("wrong fig name")
    return out

def gd(model, dataset, lr, epochs, snapshot_every=100):
    t_list = [row[0] for row in dataset]
    for epoch in range(epochs):
        L = model.loss(dataset)
        L.backward()
        L.step(lr)
        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={L.data}")
        if epoch % snapshot_every == 0 or epoch == epochs - 1:
            model.snapshots.append((epoch, model.get_graph(t_list)))

def sgd_momentum(model, dataset, lr=0.1, epochs=500, batch_size=8, beta=0.9,
                 lr_min=0.01, snapshot_every=100):
    t_list = [row[0] for row in dataset]
    for epoch in range(epochs):
        # cosine annealing: lr плавно падает от lr до lr_min
        cos = math.cos(math.pi * epoch / epochs)
        cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + cos)

        shuffled = dataset[:]
        random.shuffle(shuffled)
        total_loss = 0.0
        batches = 0
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            L = model.loss(batch)
            L.backward()
            L.momentum_step(cur_lr, beta)
            total_loss += L.data
            batches += 1
        if epoch % 10 == 0:
            print(f"epoch={epoch}, lr={cur_lr:.4f}, loss={total_loss / batches:.6f}")
        if epoch % snapshot_every == 0 or epoch == epochs - 1:
            model.snapshots.append((epoch, model.get_graph(t_list)))

def adam(model, dataset, lr=0.01, epochs=500, batch_size=8, snapshot_every=100):
    t_list = [row[0] for row in dataset]
    step = 0
    for epoch in range(epochs):
        shuffled = dataset[:]
        random.shuffle(shuffled)
        total_loss = 0.0
        batches = 0
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            step += 1
            L = model.loss(batch)
            L.backward()
            L.adam_step(lr, step)
            total_loss += L.data
            batches += 1
        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={total_loss / batches}")
        if epoch % snapshot_every == 0 or epoch == epochs - 1:
            model.snapshots.append((epoch, model.get_graph(t_list)))

def sgd(model, dataset, lr, epochs, batch_size=8, num_snapshots=20):
    t_list = [row[0] for row in dataset]
    snap_epochs = {
        i * (epochs - 1) // (num_snapshots - 1)
        for i in range(num_snapshots)
    }
    for epoch in range(epochs):
        shuffled = dataset[:]
        random.shuffle(shuffled)
        total_loss = 0.0
        batches = 0
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            L = model.loss(batch)
            L.backward()
            L.step(lr)
            total_loss += L.data
            batches += 1
        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={total_loss / batches}")
        if epoch in snap_epochs:
            model.snapshots.append((epoch, model.get_graph(t_list)))

class SimpleNN:
    def __init__(self, n0, n1, n2, n3):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.w1 = self._rand_matrix(n0, n1)
        self.b1 = self._rand_array(n1)
        self.w2 = self._rand_matrix(n1, n2)
        self.b2 = self._rand_array(n2)
        self.w3 = self._rand_matrix(n2, n3)
        self.b3 = self._rand_array(n3)
        self.snapshots = []

    def forward(self, h0):
        h1 = []
        for i in range(self.n1):
            h1i = Value(0.0)
            for j in range(self.n0):
                h1i = Value.add(h1i, Value.mul(h0[j], self.w1[j][i]))
            h1i = Value.add(h1i, self.b1[i])
            h1.append(Value.tanh(h1i))
        h2 = []
        for i in range(self.n2):
            h2i = Value(0.0)
            for j in range(self.n1):
                h2i = Value.add(h2i, Value.mul(h1[j], self.w2[j][i]))
            h2i = Value.add(h2i, self.b2[i])
            h2.append(Value.tanh(h2i))
        h3 = []
        for i in range(self.n3):
            h3i = Value(0.0)
            for j in range(self.n2):
                h3i = Value.add(h3i, Value.mul(h2[j], self.w3[j][i]))
            h3i = Value.add(h3i, self.b3[i])
            h3.append(Value.tanh(h3i))
        return h3

    def loss(self, batch):
        out = Value(0.0)
        for b in range(len(batch)):
            x = [Value(batch[b][i]) for i in range(self.n0)]
            y = [Value(batch[b][self.n0 + m]) for m in range(self.n3)]
            v = self.forward(x)
            for m in range(self.n3):
                out = Value.add(out, Value.pow(Value.sub(v[m], y[m]), 2))
        out = Value.mul(out, Value(1.0/len(batch)))
        return out

    def predict(self, t):
        h0 = [Value(t)]
        out = self.forward(h0)
        return out[0].data, out[1].data

    def get_graph(self, t_list):
        net_x, net_y = [], []
        for t in t_list:
            x, y = self.predict(t)
            net_x.append(x)
            net_y.append(y)
        return net_x, net_y

    def _rand_array(self, n):
        out = []
        for i in range(n):
            out.append(Value(random.gauss(0, 1.0 / math.sqrt(n))))
        return out

    def _rand_matrix(self, n, m):
        out = []
        for i in range(n):
            out.append([])
            for j in range(m):
                out[len(out) - 1].append(Value(random.gauss(0, 1.0 / math.sqrt(n))))
        return out
