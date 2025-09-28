import csv
import math
import logging
from typing import List, Optional

ZERO = "0"
ONE = "1"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class Node:
    """Bayesian Network Node representing a variable and its conditional probabilities."""

    def __init__(self, name: str, parents: Optional[List["Node"]] = None):
        self.name: str = name
        self.parents: List["Node"] = parents if parents else []
        self.probs: List[List[float]] = []  
        self.value: Optional[str] = None
        self.children: List["Node"] = []
        self.data_index: Optional[int] = None

    def conditional_probability(self) -> float:
        """Returns the conditional probability of the node's current value given its parents."""
        index = 0
        for i, parent in enumerate(self.parents):
            if parent.value == ZERO:
                index += 2 ** (len(self.parents) - i - 1)
        return self.probs[index][1]

    def set_children(self, nodes: List["Node"]) -> None:
        """Sets the children of this node based on parent relationships."""
        for node in nodes:
            for nP in node.parents:
                if nP == self and node not in self.children:
                    self.children.append(node)


class BayesianNetwork:
    """Encapsulates a Bayesian network and provides learning and prediction methods."""

    def __init__(self, nodes: List["Node"]):
        self.nodes: List["Node"] = nodes
        self.link_nodes()

    def link_nodes(self) -> None:
        """Converts parent names to Node references for all nodes."""
        name_to_node = {node.name: node for node in self.nodes}
        for node in self.nodes:
            node.parents = [name_to_node[p.name] if isinstance(p, Node) else name_to_node[p] for p in node.parents]

    @staticmethod
    def parse_structure(filename: str) -> List["Node"]:
        """Parses the Bayesian network structure file and returns a list of nodes."""
        nodes: List["Node"] = []
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except IOError as e:
            logging.error(f"I/O error({e.errno}): {e.strerror} : {filename}")
            raise

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            nName = parts[0].rstrip(':')
            parents = parts[1:]
            nodes.append(Node(nName, parents))
        return nodes

    @staticmethod
    def parse_data(filename: str, nodes: List["Node"]) -> List["Node"]:
        """Parses CSV data and computes CPTs for all nodes safely."""
        try:
            with open(filename, 'r', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
        except IOError as e:
            logging.error(f"I/O error({e.errno}): {e.strerror} : {filename}")
            raise

        header = data[0]

        for node in nodes:
            node_index = header.index(node.name)
            node.data_index = node_index

            if not node.parents:
                high = sum(1 for row in data[1:] if row[node_index] == ONE)
                low = sum(1 for row in data[1:] if row[node_index] == ZERO)
                total = high + low

                if total == 0:
                    node.probs = [[0.5, 0.5]]
                else:
                    node.probs = [[(low + 1) / (total + 2), (high + 1) / (total + 2)]]

            else:
                parent_indices = [header.index(p.name) for p in node.parents]
                num_rows = 2 ** len(parent_indices)
                cpt = [[1, 1] for _ in range(num_rows)]  

                for row in data[1:]:
                    idx = 0
                    for i, pi in enumerate(parent_indices):
                        if row[pi] == ZERO:
                            idx += 2 ** (len(parent_indices) - i - 1)
                    if row[node_index] == ONE:
                        cpt[idx][1] += 1
                    elif row[node_index] == ZERO:
                        cpt[idx][0] += 1

                node.probs = []
                for pair in cpt:
                    s = sum(pair)
                    if s == 0:
                        node.probs.append([0.5, 0.5])
                    else:
                        node.probs.append([pair[0] / s, pair[1] / s])

        return nodes

    @staticmethod
    def write_nodes_cpt(nodes: List["Node"], filename: str = "output.txt") -> None:
        """Writes CPTs for all nodes to a file."""
        with open(filename, 'w') as f:
            for node in nodes:
                f.write(f"{node.name}:\n")
                if not node.parents:
                    f.write(f"0, {node.probs[0][0]}\n1, {node.probs[0][1]}\n")
                else:
                    parent_names = " ".join(p.name for p in node.parents)
                    f.write(f"{parent_names}\n")
                    for i, prob_pair in enumerate(node.probs):
                        bits = bin(i)[2:].zfill(len(node.parents))
                        bits_str = ", ".join(bits)
                        f.write(f"{bits_str}, {prob_pair[1]}\n")
