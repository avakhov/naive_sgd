import random
import math

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

    def _rand_array(self, n):
        out = []
        for i in range(n):
            out.append(random.gauss(0, 1))
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
print(n.forward([0.1]))
