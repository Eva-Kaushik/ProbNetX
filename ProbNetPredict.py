#!/usr/bin/env python3
"""
ProbNetPredict.py

Predicts missing values (e.g., spam) in a CSV dataset using a Bayesian network.
Fixes parent string issue by converting all parents to Node objects before learning CPTs.
"""

import logging
import csv
from BayesNetMod import Node, BayesianNetwork, ZERO, ONE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def convert_parents_to_nodes(nodes):
    """Convert parent strings to Node objects (required for parse_data)."""
    name_to_node = {node.name: node for node in nodes}
    for node in nodes:
        new_parents = []
        for p in node.parents:
            if isinstance(p, Node):
                new_parents.append(p)
            elif isinstance(p, str) and p in name_to_node:
                new_parents.append(name_to_node[p])
        node.parents = new_parents
        node.children = []
        for other in nodes:
            if node in [name_to_node.get(p) for p in other.parents if isinstance(p, str) or isinstance(p, Node)]:
                node.children.append(other)


def markov_blanket(query_node: Node) -> float:
    """Compute probability that query_node = 1 using its Markov blanket."""
    p_vector = [0.0, 0.0]
    for value in [ZERO, ONE]:
        query_node.value = value
        p = query_node.conditional_probability()
        p_children = 1.0
        for child in query_node.children:
            if child.value == ONE:
                p_children *= child.conditional_probability()
            elif child.value == ZERO:
                p_children *= 1 - child.conditional_probability()
        if value == ZERO:
            p_vector[0] = p * p_children
        else:
            p_vector[1] = p * p_children

    total = p_vector[0] + p_vector[1]
    return 0.5 if total == 0 else p_vector[1] / total


def predict_missing(data_file: str, nodes: list, output_file="completedTest.csv"):
    """Predict missing values in CSV and write output."""
    with open(data_file, "r", newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    header = data[0]

    for node in nodes:
        if node.name in header:
            node.data_index = header.index(node.name)
        else:
            node.data_index = None

    missing_index = None
    for i, col in enumerate(header):
        if col == "?":
            missing_index = i
            break
    if missing_index is None:
        missing_index = -1

    for row in data[1:]:
        for node in nodes:
            if node.data_index is not None:
                val = row[node.data_index]
                node.value = val if val in [ZERO, ONE] else None
            else:
                node.value = None

        if missing_index >= 0:
            query_node_name = header[missing_index]
            query_node = next((n for n in nodes if n.name == query_node_name), None)
            if query_node:
                row[missing_index] = ONE if markov_blanket(query_node) >= 0.5 else ZERO

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    logging.info(f"Predictions completed. Output written to '{output_file}'.")


def main():
    structure_file = "naive_bayes.str"
    train_file = "spam_test.csv"
    test_file = "spam_test.csv"

    logging.info(f"Parsing Bayesian network structure from '{structure_file}'...")
    nodes = BayesianNetwork.parse_structure(structure_file)

    convert_parents_to_nodes(nodes)

    logging.info(f"Learning CPTs from training data '{train_file}' (safe mode)...")
    nodes = BayesianNetwork.parse_data(train_file, nodes)

    logging.info(f"Predicting missing values in '{test_file}'...")
    predict_missing(test_file, nodes)


if __name__ == "__main__":
    main()
