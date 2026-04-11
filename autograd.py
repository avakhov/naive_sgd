import math

class Value:
    def __init__(self, data, _prev=[]):
        self.data = data
        self.grad = 0.0
        self._prev = _prev
        self._backward = lambda: None
        self.m = 0.0  # Adam first moment / momentum velocity
        self.v = 0.0  # Adam second moment

    def __repr__(self):
        return f"V({self.data},{self.grad})"

    @staticmethod
    def mul(a, b):
        out = Value(a.data * b.data, _prev=[a, b])
        def _backward():
            a.grad += b.data * out.grad
            b.grad += a.data * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def add(a, b):
        out = Value(a.data + b.data, _prev=[a, b])
        def _backward():
            a.grad += out.grad
            b.grad += out.grad
        out._backward = _backward
        return out

    @staticmethod
    def sub(a, b):
        out = Value(a.data - b.data, _prev=[a, b])
        def _backward():
            a.grad += out.grad
            b.grad -= out.grad
        out._backward = _backward
        return out

    @staticmethod
    def pow(a, n):
        out = Value(pow(a.data, n), _prev=[a])
        def _backward():
            a.grad += out.grad*n*pow(a.data, n-1)
        out._backward = _backward
        return out

    @staticmethod
    def tanh(a):
        out = Value(math.tanh(a.data), _prev=[a])
        def _backward():
            a.grad += out.grad*(1 - math.tanh(a.data)**2)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def step(self, lr):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        for node in topo:
            if not node._prev:
                node.data -= lr * node.grad
                node.grad = 0.0
            else:
                node._backward = lambda: None
                node._prev = []

    def momentum_step(self, lr, beta=0.9):
        """SGD with momentum. beta=0 degenerates to plain SGD."""
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        for node in topo:
            if not node._prev:
                node.m = beta * node.m + node.grad
                node.data -= lr * node.m
                node.grad = 0.0
            else:
                node._backward = lambda: None
                node._prev = []

    def adam_step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer step. t = global step counter (starts at 1)."""
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        for node in topo:
            if not node._prev:
                node.m = beta1 * node.m + (1 - beta1) * node.grad
                node.v = beta2 * node.v + (1 - beta2) * node.grad ** 2
                m_hat = node.m / (1 - beta1 ** t)
                v_hat = node.v / (1 - beta2 ** t)
                node.data -= lr * m_hat / (v_hat ** 0.5 + eps)
                node.grad = 0.0
            else:
                node._backward = lambda: None
                node._prev = []
