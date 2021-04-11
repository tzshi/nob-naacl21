# -*- coding: utf-8 -*-

import json
from .graph import Node, Tree

def read_data(filename):
    with open(filename) as f:
        data = json.load(f)[:]

    ret = []
    for d in data:
        sent, spans, pos = d

        span_with_labels = [(l, i, j - 1) for l, i, j in spans]
        spans = [(i, j - 1) for l, i, j in spans]

        nodes = []
        for w, p in zip(sent, pos):
            nodes.append(Node(w, p))

        graph = Tree(nodes, spans)
        graph.span_with_labels = span_with_labels
        ret.append(graph)

    print(len(ret), "sentences")
    return ret
