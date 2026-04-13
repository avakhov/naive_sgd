import random
import math
from figures import points

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
            L = model.train(batch, lr)
            total_loss += L.data
            batches += 1
        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={total_loss / batches}")
        if epoch in snap_epochs:
            model.snapshots.append((epoch, model.get_graph(t_list)))

class SimpleNN:
    def __init__(self, n0, n1, n2, n3, sigma=math.tanh):
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
            h1i = 0.0
            for j in range(self.n0):
                h1i += h0[j]*self.w1[j][i]
            h1i += self.b1[i]
            h1.append(self.sigma(h1i))
        h2 = []
        for i in range(self.n2):
            h2i = 0.0
            for j in range(self.n1):
                h2i += h1[j]*self.w2[j][i]
            h2i += self.b2[i]
            h2.append(self.sigma(h2i))
        h3 = []
        for i in range(self.n3):
            h3i = 0.0
            for j in range(self.n2):
                h3i += h2[j]*self.w3[j][i]
            h3i += self.b3[i]
            h3.append(self.sigma(h3i))
        return h3

    def loss(self, batch):
        out = 0.0
        for b in range(len(batch)):
            x = [batch[b][i] for i in range(self.n0)]
            y = [batch[b][self.n0 + m] for m in range(self.n3)]
            v = self.forward(x)
            for m in range(self.n3):
                out += (v[m] - y[m])**2
        out /= len(batch)
        return out

    def train(self, batch, lr):
        dL_w1 = self._zero_matrix(n0, n1)
        dL_b1 = self._zero_array(n1)
        dL_w2 = self._zero_matrix(n1, n2)
        dL_b2 = self._zero_array(n2)
        dL_w3 = self._zero_matrix(n2, n3)
        dL_b3 = self._zero_array(n3)
        for b in range(len(batch)):
            x = [batch[b][i] for i in range(self.n0)]
            y = [batch[b][self.n0 + m] for m in range(self.n3)]
            # TODO: calc dL_b3
        L = 0.0
        return L

    def get_graph(self, t_list):
        net_x, net_y = [], []
        for t in t_list:
            x, y = self.forward([t])
            net_x.append(x)
            net_y.append(y)
        return net_x, net_y

    def _zero_array(self, n):
        out = []
        for i in range(n):
            out.append(0.0)
        return out

    def _zero_matrix(self, n, m):
        out = []
        for i in range(n):
            out.append([])
            for j in range(m):
                out[len(out) - 1].append(0.0)
        return out

    def _rand_array(self, n):
        out = []
        for i in range(n):
            out.append(random.gauss(0, 1.0 / math.sqrt(n)))
        return out

    def _rand_matrix(self, n, m):
        out = []
        for i in range(n):
            out.append([])
            for j in range(m):
                out[len(out) - 1].append(random.gauss(0, 1.0 / math.sqrt(n)))
        return out
