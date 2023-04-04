import argparse

import makemore
from micrograd import MLNN, Value


def value_tests():
    # Value tests
    operand_1 = Value(1.0)
    for operand_2 in [int(1), float(1), complex(1, 1), Value(1)]:
        for op in ["+", "-", "*", "/"]:
            assert isinstance(eval(f"operand_1 {op} operand_2"), Value)
            assert isinstance(eval(f"operand_2 {op} operand_1"), Value)
    val = Value(2.0)
    assert isinstance(val.pow(3), Value)
    assert isinstance(val.tanh(), Value)

    # Maths tests
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)
    d = Value(4.0)

    ad = a * d
    bc = b * c
    ad_bc = ad - bc
    ad_bc_tanh = ad_bc.tanh()
    ad_bc_pow_2 = ad_bc.pow(2)
    loss = ad_bc_tanh.square_norm()

    for comparison, val, expected in [
        ("==", ad, 4.0),
        ("==", bc, 6.0),
        ("==", ad_bc, -2.0),
        ("==", ad_bc_pow_2, 4.0),
        ("<", ad_bc_tanh, -0.9),
    ]:
        expression = f"{val.data} {comparison} {expected}"
        assert eval(expression), (val, comparison, expected)

    # Gradient tests
    ad_bc_tanh.grad = 1.0
    assert all(val.grad == 0.0 for val in [a, b, c, d, ad, bc, ad_bc, ad_bc_pow_2])
    ad_bc_tanh.backward()
    ad_bc_tanh
    ad_bc


def run_tests():
    for check_runner in [value_tests]:
        try:
            check_runner()
        except AssertionError as ae:
            print("Encountered assertion-error:", ae)
    print("All checks passed!")




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


def ngram() -> None:
    pass


def main(
    tests: bool,
    micrograd: bool,
    makemore: bool,  # Think about argparse namespaces
):
    if tests:
        run_tests()
    if micrograd:
        train_nn()
    if makemore:
        ngram()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("gpt", description="Try out ML code")
    parser.add_argument("--tests", action="store_true", default=False, help="Run tests")
    parser.add_argument("--micrograd", action="store_true", default=False, help="Train a simple NN")
    parser.add_argument("--makemore", action="store_true", default=False, help="Generate names by training on n-grams")
    args = parser.parse_args()
    main(
        args.tests,
        args.micrograd,
        args.makemore,
    )
