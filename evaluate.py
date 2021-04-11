# -*- coding: utf-8 -*-

from collections import Counter
from nob.parser import NOBParser
from nob.data import DataProcessor, InfiniteDataLoader, DataCollate
import json

PUNC = {'.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '#', '$'}

# Evaluation setting ignores punctuations
def process_span(l, r, graph):
    for i in range(l, r + 1):
        if graph.nodes[i].pos not in PUNC:
            l = i
            break
    for i in range(r, l - 1, -1):
        if graph.nodes[i].pos not in PUNC:
            r = i
            break
    return l, r

# Evaluation setting ignores punctuations and trivial spans
def process_spans(spans, graph):
    ret = set()
    for l, r in spans:
        for i in range(l, r + 1):
            if graph.nodes[i].pos not in PUNC:
                l = i
                break
        for i in range(r, l - 1, -1):
            if graph.nodes[i].pos not in PUNC:
                r = i
                break
        # r-l = 0 means length-1 trivial span
        if r - l > 0:
            ret.add((l, r))

    if len(ret):
        # max_span covers the whole sentence
        max_span = max(ret, key=lambda x: x[1] - x[0])
        ret = ret - {max_span}

    return ret

def calc_portion(correct, total):
    if total == 0:
        return 1.
    else:
        return correct / total

def calc_f1(precision, recall):
    if precision == 0. or recall == 0.:
        return 0.
    else:
        return 2./(1./precision + 1./recall)

parser = NOBParser()

setting = "qasrl_strict"

parser.load_model(f"./models/{setting}/model", batch_size=16)
parser._model.eval()

test_data = DataProcessor("./data/ptb-test.json", parser, parser._model)

parser.evaluate(test_data)

corpus_correct = 0
corpus_precision = 0
corpus_recall = 0
sent_f1s = []

recall_total = Counter()
recall_found = Counter()

for g in test_data.graphs:
    if len([x for x in g.nodes if x.pos not in PUNC]) <= 1:
        continue

    gold_spans = process_spans(g.spans, g)
    pred_spans = process_spans(g.pred_spans, g)

    corpus_correct += len(gold_spans & pred_spans)
    corpus_precision += len(pred_spans)
    corpus_recall += len(gold_spans)

    sent_precision = calc_portion(len(gold_spans & pred_spans), len(pred_spans))
    sent_recall = calc_portion(len(gold_spans & pred_spans), len(gold_spans))
    sent_f1 = calc_f1(sent_precision, sent_recall)
    sent_f1s.append(sent_f1)

    span_with_labels = g.span_with_labels
    for l, i, j in span_with_labels:
        l = l.split("-")[0].strip("=1234")
        i, j = process_span(i, j, g)

        # exclude potential trivial spans
        if (i, j) in gold_spans:
            recall_total[l] += 1
            if (i, j) in pred_spans:
                recall_found[l] += 1

corpus_precision = calc_portion(corpus_correct, corpus_precision)
corpus_recall = calc_portion(corpus_correct, corpus_recall)
corpus_f1 = calc_f1(corpus_precision, corpus_recall)
print("Corpus-level F1", corpus_f1)
print("Sentence-level F1", sum(sent_f1s) / len(sent_f1s))

print("Per-label recalls:")
for l, c in recall_total.most_common():
    print(l, f"{recall_found[l]}/{c}={recall_found[l]/c}")
