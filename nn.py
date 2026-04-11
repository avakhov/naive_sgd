import random
import math

def points(fig, n):
    ta = [t / n for t in range(n + 1)]
    out = []
    if fig == "heart":
        for t in ta:
            angle = t * 2 * math.pi
            x = 16 * math.sin(angle) ** 3
            y = 13 * math.cos(angle) - 5 * math.cos(2 * angle) - 2 * math.cos(3 * angle) - math.cos(4 * angle)
            out.append([t, x, y])
    else:
        raise ValueError("wrong fig name")
    return out

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
        self.sigma = math.tanh
        self.deriv = lambda x: 1 - math.tanh(x)**2

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
        for i in range(len(batch)):
            x = []
            for k in range(self.n0):
                x.append(batch[i][k])
            y = []
            for m in range(self.n3):
                y.append(batch[i][self.n0 + m])
            v = self.forward(x)
            for m in range(self.n3):
                out += (v[m] - y[m])**2
        return out / len(batch)

    def train_step(self, batch, lr):
        dL_dw1 = self._zero_matrix(self.n0, self.n1)
        dL_db1 = self._zero_array(self.n1)
        dL_dw2 = self._zero_matrix(self.n1, self.n2)
        dL_db2 = self._zero_array(self.n2)
        dL_dw3 = self._zero_matrix(self.n2, self.n3)
        dL_db3 = self._zero_array(self.n3)
        out = 0.0
        for i in range(len(batch)):
            x = []
            for k in range(self.n0):
                x.append(batch[i][k])
            y = []
            for m in range(self.n3):
                y.append(batch[i][self.n0 + m])
            h0 = x
            h1 = []
            for i in range(self.n1):
                h1i = 0.0
                for j in range(self.n0):
                    h1i += x[j]*self.w1[j][i]
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
            v = h3
            for m in range(self.n3):
                out = out + (v[m] - y[m])**2
        out = out / len(batch)
        print(out)

    def _zero_array(self, n):
        out = []
        for i in range(n):
            out.append(0.0)
        return out

    def _rand_array(self, n):
        out = []
        for i in range(n):
            out.append(random.gauss(0, 1))
        return out

    def _zero_matrix(self, n, m):
        out = []
        for i in range(n):
            out.append([])
            for j in range(m):
                out[len(out) - 1].append(0.0)
        return out

    def _rand_matrix(self, n, m):
        out = []
        for i in range(n):
            out.append([])
            for j in range(m):
                out[len(out) - 1].append(random.gauss(0, 1))
        return out

random.seed(123)
n = SimpleNN(n0=1, n1=20, n2=15, n3=2)
data = points("heart", 10)
n.train_step(data, 0.1)
