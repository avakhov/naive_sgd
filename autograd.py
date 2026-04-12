import math

class Value:
    def __init__(self, data, _prev=[]):
        self.data = data
        self.grad = 0.0
        self._prev = _prev
        self._backward = lambda: None

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
    def relu(a):
        out = Value(max(0.0, a.data), _prev=[a])
        def _backward():
            a.grad += out.grad * (1.0 if a.data > 0 else 0.0)
        out._backward = _backward
        return out

    @staticmethod
    def elu(a, alpha=1.0):
        v = a.data if a.data > 0 else alpha * (math.exp(a.data) - 1)
        out = Value(v, _prev=[a])
        def _backward():
            a.grad += out.grad * (1.0 if a.data > 0 else alpha * math.exp(a.data))
        out._backward = _backward
        return out

    @staticmethod
    def silu(a):
        sig = 1.0 / (1.0 + math.exp(-a.data))
        out = Value(a.data * sig, _prev=[a])
        def _backward():
            sig = 1.0 / (1.0 + math.exp(-a.data))
            a.grad += out.grad * (sig + a.data * sig * (1.0 - sig))
        out._backward = _backward
        return out

    @staticmethod
    def sin(a):
        out = Value(math.sin(a.data), _prev=[a])
        def _backward():
            a.grad += out.grad * math.cos(a.data)
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
