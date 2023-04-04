# L*earning* AI *and* M*achine-learning*
Follow the courses of Andrej Karpathy on LLMs


## License
MIT


## Part -1: About

The entrypoint for the whole repo is the `main.py` script. Running it with different flags will run different parts of the codebase. Use it to run tests:
```bash
$ python3 main.py --tests
```


## Part 0: The spelled-out intro to neural networks and backpropagation: building micrograd
[Video](https://youtu.be/VMj-3S1tku0) | [Andrej Karpathy's repo](https://github.com/karpathy/micrograd)

Build from first principles a MLNN.
- Start by creating a `Value` class with inbuilt forward- and backward-propagation for given mathematical operations
- Use `Value` in a neural-net structure
- Train a small neural-net on data

```bash
$ python3 main.py --micrograd
```


## Part 1: The spelled-out intro to language modeling: building makemore
[Video](https://youtu.be/PaCmpygFfXo) | [Andrej Karpathy's repo](https://github.com/karpathy/makemore)

Use n-gram statistical data from a large dataset of names to generate new real-looking names.

```bash
$ python3 main.py --makemore
```
