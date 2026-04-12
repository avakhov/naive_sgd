import random
import math
from autograd import Value
from figures import points

def gd(model, dataset, lr, epochs, num_snapshots=20, snap_points=100):
    t_list = [i / snap_points for i in range(snap_points + 1)]
    snap_epochs = {
        i * (epochs - 1) // (num_snapshots - 1)
        for i in range(num_snapshots)
    }
    for epoch in range(epochs):
        L = model.loss(dataset)
        L.backward()
        L.step(lr)
        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={L.data}")
        if epoch in snap_epochs:
            model.snapshots.append((epoch, model.get_graph(t_list)))


def sgd(model, dataset, lr, epochs, batch_size=32, num_snapshots=20, snap_points=100):
    t_list = [i / snap_points for i in range(snap_points + 1)]
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
    def __init__(self, n0, n1, n2, n3, sigma=Value.tanh):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.sigma = sigma
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
            h1.append(self.sigma(h1i))
        h2 = []
        for i in range(self.n2):
            h2i = Value(0.0)
            for j in range(self.n1):
                h2i = Value.add(h2i, Value.mul(h1[j], self.w2[j][i]))
            h2i = Value.add(h2i, self.b2[i])
            h2.append(self.sigma(h2i))
        h3 = []
        for i in range(self.n3):
            h3i = Value(0.0)
            for j in range(self.n2):
                h3i = Value.add(h3i, Value.mul(h2[j], self.w3[j][i]))
            h3i = Value.add(h3i, self.b3[i])
            h3.append(self.sigma(h3i))
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
