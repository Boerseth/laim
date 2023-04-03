import argparse

import makemore
from micrograd import MLNN, Value


def show_output(output: list[Value]) -> None:
    print(" > ", [f"{y:.4f}" for y in output])


def train_nn() -> None:
    training_input = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    training_output = [1.0, -1.0, -1.0, 1.0]
    input_values = [[Value(x) for x in data_i] for data_i in training_input]
    output_values = [Value(y) for y in training_output]

    network = MLNN([4, 4, 1])
    parameters = list(network.parameters())

    loss = sum((network(x) - y).pow(2) for x, y in zip(input_values, output_values))
    print(f"Goal:     ")
    show_output(training_output)
    print(f"Untrained:")
    show_output([network(x).data for x in training_input])
    print(f"Loss before training: {loss.data}")
    print("training...")
    for _ in range(300):
        loss.grad = 1.0
        loss.backward()
        for p in parameters:
            p.descend(0.1)
        for p in parameters:
            p.forward()
    print(f"Loss after training: {loss.data}")
    print(f"Trained:  ")
    show_output([network(x).data for x in training_input])


def main(
    micrograd: bool,
):
    if micrograd:
        train_nn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("gpt", description="Try out ML code")
    parser.add_argument("--micrograd", action="store_true", default=False, help="Train a simple NN")
    args = parser.parse_args()
    main(
        args.micrograd,
    )
