# -*- coding: utf-8 -*-

import numpy as np


class Node(object):

    def __init__(self, word, pos):
        self.word = word
        self.pos = pos

    def __repr__(self):
        return "%s-%s" % (self.word, self.pos)


class Tree(object):

    def __init__(self, nodes, spans=None):
        self.nodes = np.array(nodes)
        if spans:
            self.spans = spans
        else:
            self.spans = []

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.nodes = self.nodes
        result.spans = deepcopy(self.spans)

        return result

    def cleaned(self):
        return Tree(self.nodes[:])
