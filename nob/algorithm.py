# -*- coding: utf-8 -*-

import numpy as np

# scores is a square matrix
def parse_cky(scores):
    nr, nc = np.shape(scores)

    chart = np.zeros((nr, nc)) - np.inf
    backtrack = np.zeros((nr, nc), dtype=int)

    # initialize CKY chart, populate single leave nodes
    for i in range(nr):
        chart[i, i] = scores[i, i]

    # Loop from smaller items to larger items.
    for k in range(1, nr):
        for i in range(nr - k):
            j = i + k

            vals = chart[i, i:j] + chart[i+1:j+1, j] + scores[i, j]
            chart[i, j] = np.max(vals)
            backtrack[i, j] = i + np.argmax(vals)

    spans = []
    backtrack_cky(backtrack, 0, nr - 1, spans)

    return spans


def backtrack_cky(backtrack, i, j, spans):
    if i == j:
        return
    spans.append((i, j))
    k = int(backtrack[i, j])
    backtrack_cky(backtrack, i, k, spans)
    backtrack_cky(backtrack, k + 1, j, spans)
