# -*- coding: utf-8 -*-

import json
import sys
import os
import time
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn as nn
import fire
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam

from .modules import ChartParser
from .features import WordSequence
from .io import read_data
from .data import DataProcessor, DataCollate, InfiniteDataLoader


class NOBParser:

    def __init__(self, **kwargs):
        pass

    def create_parser(self, **kwargs):
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
            sys.stdout.flush()

        self._args = kwargs

        self._gpu = kwargs.get("gpu", True)

        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)
        self._epsilon = kwargs.get("epsilon", 1e-12)

        self._clip = kwargs.get("clip", 1.)

        self._batch_size = kwargs.get("batch_size", 8)
        self._warmup = kwargs.get("warmup", 2000)

        self._chart_dims = kwargs.get("chart_dims", 256)
        self._chart_dropout = kwargs.get("chart_dropout", 0.33)
        self._chart_mode = kwargs.get("chart_mode", "strict")

        self._bert_model = kwargs.get("bert_model", "bert-base-uncased")

        self.init_model()

        return self

    def save_model(self, filename):
        print("Saving model to", filename)
        with open(filename + ".params", "w") as f:
            json.dump(self._args, f)
        torch.save(self._model.state_dict(), filename + '.model')

    def load_model(self, filename, **kwargs):
        print("Loading model from", filename)
        with open(filename + ".params", "r") as f:
            args = json.load(f)
            args.update(kwargs)
            self.create_parser(**args)
        self._model.load_state_dict(torch.load(filename + ".model"))
        return self

    def init_model(self):
        self._seqrep = WordSequence(self)

        self._chart_parser = ChartParser(self, self._chart_dims, dropout=self._chart_dropout, mode=self._chart_mode)
        self._modules = [self._chart_parser]
        self._model = nn.ModuleList([self._seqrep] + self._modules)

        if self._gpu:
            self._model.cuda()
        return self

    def train(self, filename, train_steps=100, save_prefix=None, **kwargs):
        train_graphs = DataProcessor(filename, self, self._model)
        train_loader = InfiniteDataLoader(train_graphs, batch_size=self._batch_size, shuffle=True, num_workers=1, collate_fn=DataCollate(self, train=True))

        optimizer = Adam(self._model.parameters(), lr=self._learning_rate, betas=(self._beta1, self._beta2), eps=self._epsilon)

        print("Model")
        for param_tensor in self._model.state_dict():
            print(param_tensor, "\t", self._model.state_dict()[param_tensor].size())
        print("Opt")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        t0 = time.time()
        results, eloss = defaultdict(float), 0.

        update_steps = 0
        cur_update_steps = 0
        for batch_i, batch in enumerate(train_loader):
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw":
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            self._model.train()
            self._model.zero_grad()

            loss = []

            seq_features = self._seqrep(batch)

            for module in self._modules:
                batch_label = module.batch_label(batch)
                l, pred = module.calculate_loss(seq_features, batch)

                if l is not None:
                    loss.append(l)

            if len(loss):
                loss = sum(loss)
                loss.backward()

                update_steps += 1
                cur_update_steps += 1
                if update_steps <= self._warmup:
                    for param in optimizer.param_groups:
                        param['lr'] = min(self._learning_rate, self._learning_rate * update_steps / self._warmup)

                nn.utils.clip_grad_norm_(self._model.parameters(), self._clip)
                optimizer.step()

            if batch_i and batch_i % 100 == 0:
                print(batch_i // 100, "{:.2f}s".format(time.time() - t0), "margin violations", cur_update_steps)
                sys.stdout.flush()
                cur_update_steps = 0
                t0 = time.time()

            if batch_i >= train_steps:
                break

        if save_prefix:
            self.save_model("{}model".format(save_prefix))

        return self

    def evaluate(self, data, output_file=None):
        self._model.eval()
        start_time = time.time()
        dev_loader = DataLoader(data, batch_size=self._batch_size, shuffle=False, num_workers=1, collate_fn=DataCollate(self, train=False))

        with torch.no_grad():
            for batch in dev_loader:
                if self._gpu:
                    for k in batch:
                        if k != "graphidx" and k != "raw":
                            batch[k] = batch[k].cuda()

                mask = batch["mask"]

                seq_features = self._seqrep(batch)

                for module in self._modules:
                    batch_label = module.batch_label(batch)
                    pred = module(self, seq_features, batch)

                if "pred_lefts" in batch:
                    pred_lefts = batch["pred_lefts"].cpu().data.numpy()
                    pred_rights = batch["pred_rights"].cpu().data.numpy()
                    for idx, l, r in zip(batch["graphidx"], pred_lefts, pred_rights):
                        g = data.graphs[idx]
                        g.pred_spans = [(ll, rr) for ll, rr in zip(l, r) if rr - ll > 0]

        decode_time = time.time() - start_time
        print("Speed", len(data) / decode_time)


    def finish(self, **kwargs):
        print()
        sys.stdout.flush()

if __name__ == '__main__':
    fire.Fire(NOBParser)
