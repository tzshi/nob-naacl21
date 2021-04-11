# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import BilinearMatrixAttention
from .algorithm import parse_cky


class ChartParser(nn.Module):

    name = "Chart"

    def __init__(self, parser, hidden_size, dropout=0., mode="strict"):
        super(ChartParser, self).__init__()
        print("build chart parser ...", self.__class__.name)

        self.left_mlp = nn.Sequential(
            nn.Linear(parser._seqrep_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.right_mlp = nn.Sequential(
            nn.Linear(parser._seqrep_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.scorer = BilinearMatrixAttention(hidden_size, hidden_size, True)
        self.mode = mode

    def calculate_loss(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]

        lefts = self.left_mlp(lstm_features)
        rights = self.right_mlp(lstm_features)
        batch_size = batch_label[0].size(0)
        seq_len = batch_label[0].size(1)

        scores = self.scorer(lefts, rights)

        mask_np = mask.data.cpu().numpy()
        scores_np = scores.data.cpu().numpy()
        g_lefts = batch_label[0].data.cpu().numpy()
        g_rights = batch_label[1].data.cpu().numpy()

        loss = []
        for i in range(batch_size):
            length = sum(mask_np[i])

            cost = np.zeros((length, length))

            if self.mode == "loose":
                for j, k in zip(g_lefts[i], g_rights[i]):
                    if j == k or j >= length or k >= length:
                        continue

                    for r in range(j, k):
                        for l in range(0, j):
                            cost[l, r] = 1.
                    for l in range(j + 1, k + 1):
                        for r in range(k + 1, length):
                            cost[l, r] = 1.

                for j, k in zip(g_lefts[i], g_rights[i]):
                    if j == k or j >= length or k >= length:
                        continue
                    cost[j, k] = 0.

            elif self.mode == "strict":
                cost += 1.
                for j, k in zip(g_lefts[i], g_rights[i]):
                    if j == k or j >= length or k >= length:
                        continue
                    cost[j, k] = 0.

            aug_spans = set(parse_cky(scores_np[i,:length,:length] + cost))
            dim_spans = set(parse_cky(scores_np[i,:length,:length] - cost))

            for j, k in aug_spans - dim_spans:
                loss.append(scores[i, j, k])
            for j, k in dim_spans - aug_spans:
                loss.append(-scores[i, j, k])

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1

        if len(loss):
            total_loss = sum(loss) / float(batch_size)
        else:
            total_loss = None

        return total_loss, (tag_seq, tag_seq)

    def forward(self, parser, lstm_features, batch):
        mask = batch["mask"]

        lefts = self.left_mlp(lstm_features)
        rights = self.right_mlp(lstm_features)
        batch_size = mask.size(0)
        seq_len = mask.size(1)

        scores = self.scorer(lefts, rights)

        mask_np = mask.data.cpu().numpy()
        scores_np = scores.data.cpu().numpy()

        ret_lefts = torch.zeros(batch_size, seq_len).long()
        ret_rights = torch.zeros(batch_size, seq_len).long()

        for i in range(batch_size):
            length = sum(mask_np[i])
            spans = set(parse_cky(scores_np[i,:length,:length]))

            for ii, (j, k) in enumerate(spans):
                ret_lefts[i, ii] = j
                ret_rights[i, ii] = k

        ret_lefts = ret_lefts.to(mask.device)
        ret_rights = ret_rights.to(mask.device)

        batch["pred_lefts"] = ret_lefts
        batch["pred_rights"] = ret_rights

        return (ret_lefts, ret_rights)

    @staticmethod
    def load_data(parser, graph):
        return {"span_lefts": [x[0] for x in graph.spans], "span_rights": [x[1] for x in graph.spans]}

    @staticmethod
    def batch_label(batch):
        return batch["span_lefts"], batch["span_rights"]
