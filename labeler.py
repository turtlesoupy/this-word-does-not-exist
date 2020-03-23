import pandas as pd
import os
import sys
import argparse
import pickle
import itertools
import random
from math import sqrt


def wilson(ups, downs):
    n = ups + downs
    if n == 0:
        return 0

    z = 1.96  # 1.44 = 85%, 1.96 = 95%
    phat = float(ups) / n
    return (phat + z * z / (2 * n) - z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    while True:
        sys.stdout.write(question + " [y/n] ")
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def do_label(datasets, limit=None, prompt=None, dataset_labels=None):
    clear = lambda: os.system("clear")

    items_to_label = list(
        itertools.chain.from_iterable([[(i, d) for d in dataset] for i, dataset in enumerate(datasets)])
    )
    random.shuffle(items_to_label)

    dataset_positives = [0] * len(datasets)
    dataset_totals = [len(d) for d in datasets]

    for dataset_idx, word in items_to_label[:limit]:
        clear()
        print(f"**Word: {word.word}**")
        print()
        print(f"**Definition**")
        print(word.definition)
        print()
        print(f"**Example**")
        print(word.example)
        print()
        response = query_yes_no(prompt or "Label: ")
        if response:
            dataset_positives[dataset_idx] += 1

    dataset_negatives = [dataset_totals[i] - dataset_positives[i] for i in range(len(datasets))]
    wilsons = [wilson(dataset_positives[i], dataset_negatives[i]) for i in range(len(datasets))]

    print(f"Dataset totals:")
    print(
        "\n".join(
            f"\t{dataset_labels[i] if dataset_labels else i} - {dataset_positives[i] / dataset_totals[i]:.3f} - +/- {abs(dataset_positives[i] / dataset_totals[i] - wilsons[i]):.3f}"
            for i in range(len(datasets))
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive labeling for datasets")
    parser.add_argument("--datasets", action="append", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt", type=str, help="The prompt for terminal")
    args = parser.parse_args()

    if len(args.datasets) < 2:
        raise RuntimeError("Require a minimum of two datasets")

    datasets = [pickle.load(open(d, "rb")) for d in args.datasets]
    do_label(datasets, limit=args.limit, prompt=args.prompt, dataset_labels=args.datasets)


if __name__ == "__main__":
    main()
