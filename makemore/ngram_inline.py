import argparse
from collections import defaultdict
from math import log
from random import random, seed
from string import ascii_lowercase as alphabet
from typing import Any

from tools import get_http, str_file_cache


seed(3)

PAD = "."
TOKENS = [PAD, *list(alphabet)]
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILENAME = DATA_URL.split("/")[-1]

@str_file_cache(DATA_FILENAME)
def get_data(force_refresh: bool = False) -> list[str]:
    return get_http(DATA_URL)


def get_name_fitness(name: str, cond_prob: dict[tuple[str, str], float], n: int) -> float:
    fitness = 1.0
    name_padded = [PAD] * (n - 1)  +  list(name) + [PAD]
    name_offsets = [name_padded[i:] for i in range(n)]
    for *first, last in zip(*name_offsets):
        fitness *= cond_prob["".join(first), last]
    return fitness ** (1 / (1 + len(name)))


def get_negative_log_likelihood(name: str, prob: dict[tuple[str, str], float], n: int) -> float:
    nll = 0.0
    name_padded = [PAD] * (n - 1) + list(name) + [PAD]
    name_offsets = [name_padded[i:] for i in range(n)]
    for *first, last in zip(*name_offsets):
        nll -= log(prob["".join(first), last])
    return nll


def generate_name(cum_cond_prob: dict[tuple[str, str], float], n: int) -> str:
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


def main(n: int, word_count: int, show_names: bool) -> None:
    names = get_data().splitlines()

    frequency = defaultdict(int)
    for name in names:
        name_padded = [PAD] * (n - 1) + list(name) + [PAD]
        name_offsets = [name_padded[i:] for i in range(n)]
        for *first, last in zip(*name_offsets):
            frequency["".join(first), last] += 1

    # P = f / N
    total = sum(frequency.values())
    prob = {key: f / total for key, f in frequency.items()}

    # P(B) = sum(P(A n B) for all A)
    marg_prob = defaultdict(float)
    for (B, _), prob_b_a in prob.items():
        marg_prob[B] += prob_b_a

    # P(A | B) = P(A n B) / P(B)
    cond_prob = defaultdict(float)
    for (B, A), prob_b_a in prob.items():
        cond_prob[B, A] = prob_b_a / marg_prob[B]

    # P(a <= A | B) = sum(P(a=A | B) for a <= A)
    cum_cond_prob = defaultdict(float)
    for B in {B for B, _ in cond_prob}:
        for A, A_prev in zip(TOKENS, [PAD, *TOKENS]):
            cum_cond_prob[B, A] = cond_prob[B, A] + cum_cond_prob[B, A_prev]

    data = []
    for _ in range(word_count):
        name = generate_name(cum_cond_prob, n)
        fitness = get_name_fitness(name, cond_prob, n)
        nll = get_negative_log_likelihood(name, cond_prob, n)
        ngram_count = len(name) + 1

        data_i = {"name": name, "fitness": fitness, "nll": nll, "ngram_count": len(name) + 1}

        if show_names:
            nll_str = f"{nll:.5f}".rjust(8, " ")
            print(f"fitness: {fitness:.5f}, nll: {nll_str}, name: {name}")
        data.append(data_i)

    name_set = set(names)
    overfit = sum(1 for d in data if d["name"] in name_set) / len(data)
    negative_log_likelihood = sum(d["nll"] for d in data)
    avg_nll = negative_log_likelihood / sum(d["ngram_count"] for d in data)
    print(f"{avg_nll=}")
    print(f"{overfit=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate names")
    parser.add_argument("-n", "--n-gram", type=int, default=2)
    parser.add_argument("-wc", "--word-count", type=int, default=10)
    parser.add_argument("-sn", "--show-names", action="store_true", default=False)
    args = parser.parse_args()
    main(args.n_gram, args.word_count, args.show_names)
