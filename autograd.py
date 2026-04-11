from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass(eq=False)
class Value:
    data: float
    grad: float = 0.0
    _prev: set["Value"] = field(default_factory=set)
    _backward: Callable[[], None] = field(default=lambda: None)
    _op: str = ""

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __hash__(self) -> int:
        return id(self)

    @staticmethod
    def _ensure_value(x: float | "Value") -> "Value":
        return x if isinstance(x, Value) else Value(float(x))

    def __add__(self, other: float | "Value") -> "Value":
        other = self._ensure_value(other)
        out = Value(self.data + other.data, _prev={self, other}, _op="+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        out = Value(-self.data, _prev={self}, _op="neg")

        def _backward() -> None:
            self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-self._ensure_value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return self._ensure_value(other) + (-self)

    def __mul__(self, other: float | "Value") -> "Value":
        other = self._ensure_value(other)
        out = Value(self.data * other.data, _prev={self, other}, _op="*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __truediv__(self, other: float | "Value") -> "Value":
        other = self._ensure_value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other: float | "Value") -> "Value":
        other = self._ensure_value(other)
        return other * (self ** -1)

    def __pow__(self, power: float) -> "Value":
        if not isinstance(power, (int, float)):
            raise TypeError("Поддерживаются только int/float степени")

        out = Value(self.data ** power, _prev={self}, _op=f"**{power}")

        def _backward() -> None:
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for node in topo:
            node.grad = 0.0


x = Value(2.0)
y = Value(3.0)
z = x * y + x ** 2 - y / 2
z.backward()
print("z =", z.data)
print("dz/dx =", x.grad)  # y + 2x = 3 + 4 = 7
print("dz/dy =", y.grad)  # x - 1/2 = 2 - 0.5 = 1.5
