import argparse
from collections import defaultdict
from random import random, seed
from string import ascii_lowercase as alphabet
from typing import Any

import torch

from tools import get_http, str_file_cache


Token = str
TokenPair = tuple[Token, Token]
Frequency = dict[TokenPair, int]
Probability = dict[TokenPair, float]
MarginalProbability = dict[Token, float]
ConditionalProbability = dict[TokenPair, float]
CumulativeConditionalProbability = dict[TokenPair, float]

PAD = "."
TOKENS = [PAD, *list(alphabet)]

seed(3)

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILENAME = DATA_URL.split("/")[-1]

@str_file_cache(DATA_FILENAME)
def get_data(force_refresh: bool = False) -> list[str]:
    return get_http(DATA_URL)


def show_extremes(
    title: str, value_map: dict[Any, int | float], count: int = 10, highest: bool = True
) -> None:
    print()
    print(f"-- {title} --")
    print(f"count: {count}, highest: {highest}")
    sorted_values = sorted(value_map.items(), key=lambda kv: kv[1], reverse=highest)
    for key, value in sorted_values[:count]:
        print(key, value)


def get_ngram_frequency(names: list[str], n: int) -> Frequency:
    frequency = defaultdict(int)
    for name in names:
        name_padded = [PAD] * (n - 1) + list(name) + [PAD]
        name_offsets = [name_padded[i:] for i in range(n)]
        for *first, last in zip(*name_offsets):
            frequency["".join(first), last] += 1
    return frequency


def get_probability(freq: Frequency) -> Probability:
    total = sum(freq.values())
    return {key: f / total for key, f in freq.items()}


def get_marginal_probability(prob: Probability) -> MarginalProbability:
    """ P(B) = sum(P(A n B) for all A) """
    marg_prob = defaultdict(float)
    for (B, _), prob_b_a in prob.items():
        marg_prob[B] += prob_b_a
    return marg_prob


def get_conditional_probability(prob: Probability, marg_prob: MarginalProbability) -> ConditionalProbability:
    """ P(A | B) = P(A n B) / P(B) """
    cond_prob = defaultdict(float)
    for (B, A), prob_b_a in prob.items():
        cond_prob[B, A] = prob_b_a / marg_prob[B]
    return cond_prob


def get_cumulative_conditional_probability(
    cond_prob: ConditionalProbability, n: int
) -> CumulativeConditionalProbability:
    cum_cond_prob = defaultdict(float)
    for B in {B for B, _ in cond_prob}:
        for A, A_prev in zip(TOKENS, [PAD, *TOKENS]):
            cum_cond_prob[B, A] = cond_prob[B, A] + cum_cond_prob[B, A_prev]
    return cum_cond_prob


def get_name_fitness(name: str, cond_prob: ConditionalProbability, n: int) -> float:
    fitness = 1.0
    padded_name = [PAD] * (n - 1)  +  list(name) + [PAD]
    for *first, last in zip(*[yield_from(padded_name[i:]) for i in range(n)]):
        fitness *= cond_prob["".join(first), last]
    return fitness ** (1 / (1 + len(name)))


def generate_name(cum_cond_prob: CumulativeConditionalProbability, n: int) -> str:
    name = PAD * (n - 1)
    while True:
        prompt = name[-(n - 1) :]
        q = random()
        # TODO: User binary search to speed it up
        for token in TOKENS:
            if q < cum_cond_prob[prompt, token]:
                name += token
                break
        if token == PAD:
            return name.strip(PAD)


def main(n) -> None:
    names = get_data().splitlines()

    frequency = torch.zeros(tuple(len(TOKENS) for _ in range(n)), dtype=torch.int32)
    for name in names:
        name_padded = [PAD] * (n - 1) + list(name) + [PAD]
        name_offsets = [name_padded[i:] for i in range(n)]
        for *first, last in zip(*name_offsets):
            frequency[*first, last] += 1



    freq = get_ngram_frequency(names, n)
    # show_extremes("Frequencies" freq)
    prob = get_probability(freq)
    marg_prob = get_marginal_probability(prob)
    cond_prob = get_conditional_probability(prob, marg_prob)
    # fitnesses = {name: get_name_fitness(name, cond_prob, n) for name in names}
    # show_extremes("Fitness", fitnesses, count=30, highest=True)
    # show_extremes("Fitness", fitnesses, count=30, highest=False)
    cum_cond_prob = get_cumulative_conditional_probability(cond_prob, n)
    for _ in range(10):
        name = generate_name(cum_cond_prob, n)
        print(f"{get_name_fitness(name, cond_prob, n):.5f}", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate names")
    parser.add_argument("N", type=int, default=2)
    args = parser.parse_args()
    main(args.N)
