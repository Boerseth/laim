from __future__ import annotations

from math import tanh


Data = int | float


class Value:
    def __init__(self, data: Data, children: tuple[Value] = ()) -> None:
        self.data = data
        self.grad = 0.0
        self.children = children

        self.parents = []
        for child in children:
            child.parents.append(self)

    def _backward(self) -> None:
        pass

    def _forward(self) -> None:
        pass

    def backward(self) -> None:
        self._backward()
        for child in self.children:
            child.backward()

    def forward(self) -> None:
        for parent in self.parents:
            parent.forward()
        self._forward()
        self.grad = 0.0

    def descend(self, speed: float) -> None:
        self.data -= self.grad * speed

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Data | Value) -> Value:
        assert isinstance(other, (int, float, Value))
        other = other if isinstance(other, Value) else Value(data=other)
        return _ValueSum(self.data + other.data, (self, other))

    def __radd__(self, other: Data) -> Value:
        return self + other

    def __mul__(self, other: Value) -> Value:
        assert isinstance(other, (int, float, Value))
        other = other if isinstance(other, Value) else Value(data=other)
        return _ValueProd(self.data * other.data, (self, other))

    def __rmul__(self, other: Data) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return _ValueProd(-self.data, (self, Value(-1)))

    def __sub__(self, other: Data | Value) -> Value:
        assert isinstance(other, (int, float, Value))
        other = other if isinstance(other, Value) else Value(data=other)
        return _ValueSum(self.data - other.data, (self, -other))

    def __rsub__(self, other: Data) -> Value:
        return self - other

    def __truediv__(self, other: Data | Value) -> Value:
        assert isinstance(other, (int, float, Value))
        other = other if isinstance(other, Value) else Value(data=other)
        return self * other.pow(-1)

    def __rtruediv__(self, other: Data) -> Value:
        return other * self.pow(-1)

    def tanh(self) -> Value:
        return _ValueTanh(tanh(self.data), self)

    def pow(self, k: int) -> Value:
        return _ValuePow(self.data**k, self, k)


class _ValueSum(Value):
    def __init__(self, data, children):
        super().__init__(data, children)
        addend_1, addend_2 = children
        self.addend_1 = addend_1
        self.addend_2 = addend_2

    def _backward(self) -> None:
        self.addend_1.grad += self.grad
        self.addend_2.grad += self.grad

    def _forward(self) -> None:
        self.data = self.addend_1.data + self.addend_2.data


class _ValueProd(Value):
    def __init__(self, data: Data, children: tuple[Value]) -> None:
        super().__init__(data, children)
        factor_1, factor_2 = self.children
        self.factor_1 = factor_1
        self.factor_2 = factor_2

    def _backward(self) -> None:
        self.factor_1.grad += self.grad * self.factor_2.data
        self.factor_2.grad += self.grad * self.factor_1.data

    def _forward(self) -> None:
        self.data = self.factor_1.data * self.factor_2.data


class _ValueTanh(Value):
    def __init__(self, data: Data, child: Value) -> None:
        super().__init__(data, (child,))
        self.child = child

    def _backward(self) -> None:
        self.child.grad += self.grad * (1 - self.data**2)

    def _forward(self) -> None:
        self.data = tanh(self.child.data)


class _ValuePow(Value):
    def __init__(self, data: Data, child: Value, k: int):
        super().__init__(data, (child,))
        self.child = child
        self.k = k

    def _backward(self) -> None:
        self.child.grad += self.grad * self.k * self.child.data ** (self.k - 1)

    def _forward(self) -> None:
        self.data = self.child.data**self.k
