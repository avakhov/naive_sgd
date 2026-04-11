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
