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

def gd(model, batch, lr):
    for i in range(1000):
        L = model.loss(batch)
        L.backward()
        L.step(lr)
        print(L.data)


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

    def _rand_array(self, n):
        out = []
        for i in range(n):
            out.append(Value(random.gauss(0, 1)))
        return out

    def _rand_matrix(self, n, m):
        out = []
        for i in range(n):
            out.append([])
            for j in range(m):
                out[len(out) - 1].append(Value(random.gauss(0, 1)))
        return out

random.seed(123)
n = SimpleNN(n0=1, n1=20, n2=15, n3=2)
data = points("heart", 50)
gd(n, data, 0.1)
