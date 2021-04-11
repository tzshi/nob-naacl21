# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .io import read_data


class DataProcessor(Dataset):

    def __init__(self, filename, parser, modules):
        data = read_data(filename)
        print("Read", filename, len(data), "trees", sum([len(x.nodes) - 1 for x in data]), "words")
        self.data = [{"graphidx": i} for i, d in enumerate(data)]
        self.graphs = data
        for m in modules:
            for d, d_ in zip(self.data, data):
                d.update(m.load_data(parser, d_))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataCollate:

    def __init__(self, parser, train=True):
        self.parser = parser
        self.train = train

    def __call__(self, data):
        ret = {}
        batch_size = len(data)
        keywords = set(data[0].keys()) - {"graphidx", "raw"}

        graphidx = [d["graphidx"] for d in data]
        raw = [d["raw"] for d in data]

        word_seq_lengths = torch.LongTensor(list(map(len, raw)))
        max_seq_len = word_seq_lengths.max().item()

        mask = torch.zeros((batch_size, max_seq_len)).byte()
        for idx, seqlen in enumerate(word_seq_lengths):
            seqlen = seqlen.item()
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)

        for keyword in keywords:
            label_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
            labels = [d[keyword] for d in data]
            for idx, (label, seqlen) in enumerate(zip(labels, word_seq_lengths)):
                label_seq_tensor[idx, :min(seqlen, len(label))] = torch.LongTensor(label[:seqlen])
            ret[keyword] = label_seq_tensor

        ret.update({
            "graphidx": graphidx,
            "raw": raw,
            "mask": mask,
        })

        return ret

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
