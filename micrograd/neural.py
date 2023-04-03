from random import uniform
from typing import Any, Callable, Iterator

from micrograd.value import Value


class Neuron:
    def __init__(self, dim_in: int) -> None:
        self.weights = [Value(uniform(-1, 1)) for _ in range(dim_in)]
        self.b = Value(uniform(-1, 1))

    def __call__(self, axon: list[Value]) -> Value:
        return sum((x * w for x, w in zip(axon, self.weights)), self.b).tanh()

    def parameters(self) -> Iterator[Value]:
        yield self.b
        yield from self.weights


class Layer:
    def __init__(self, dim_in: int, dim_out: int) -> None:
        self.neurons = [Neuron(dim_in) for _ in range(dim_out)]

    def __call__(self, axon: list[Value]) -> list[Value]:
        return [neuron(axon) for neuron in self.neurons]

    def parameters(self) -> Iterator[Value]:
        for neuron in self.neurons:
            yield from neuron.parameters()


def _apply_pipe(pipe: list[Callable[[Any], Any]], argument: Any) -> Any:
    if not pipe:
        return argument
    first, *rest = pipe
    return _apply_pipe(first(argument), rest)


class MultiLayeredNeuralNetwork:
    def __init__(self, dim_layers: list[int]) -> None:
        self.layers = [Layer(di, do) for di, do in zip(dim_layers, dim_layers[1:])]

    def __call__(self, axon: list[Value]) -> Value | list[Value]:
        out = _apply_pipe(self.layers, axon)
        return out if len(out) != 1 else out.pop()

    def parameters(self) -> Iterator[Value]:
        for layer in self.layers:
            yield from layer.parameters()
