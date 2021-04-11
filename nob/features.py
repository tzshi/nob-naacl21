# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .utils import BERT_TOKEN_MAPPING, from_numpy


class WordSequence(nn.Module):

    def __init__(self, parser):
        super(WordSequence, self).__init__()
        print("build feature extractor...")

        self.bert_tokenizer = AutoTokenizer.from_pretrained(parser._bert_model)
        self.bert_model = AutoModel.from_pretrained(parser._bert_model)

        parser._seqrep_dims = self.bert_model.config.hidden_size

    def forward(self, batch):
        mask = batch["mask"].transpose(1,0)
        raw = batch["raw"]

        seq_max_len = max([len(x) for x in raw])

        all_input_ids = np.zeros((len(raw), 2048), dtype=int)
        all_input_mask = np.zeros((len(raw), 2048), dtype=int)
        all_word_end_mask = np.zeros((len(raw), 2048), dtype=int)

        subword_max_len = 0

        for snum, sentence in enumerate(raw):
            tokens = []
            word_end_mask = []

            tokens.append("[CLS]")
            word_end_mask.append(0)

            cleaned_words = []
            for word in sentence:
                word = BERT_TOKEN_MAPPING.get(word, word)
                cleaned_words.append(word)

            for word in cleaned_words:
                word_tokens = self.bert_tokenizer.tokenize(word)

                if len(word_tokens) == 0:
                    word_tokens = ['[MASK]']

                for _ in range(len(word_tokens)):
                    word_end_mask.append(0)

                word_end_mask[-1] = 1

                tokens.extend(word_tokens)

            tokens.append("[SEP]")

            for i in range(seq_max_len - len(sentence)):
                word_end_mask.append(1)

            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            subword_max_len = max(subword_max_len, len(word_end_mask) + 1)

            all_input_ids[snum, :len(input_ids)] = input_ids
            all_input_mask[snum, :len(input_mask)] = input_mask
            all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
        all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))
        features, _ = self.bert_model(all_input_ids, attention_mask=all_input_mask, return_dict=False)

        features_packed = features.masked_select(all_word_end_mask.bool().unsqueeze(-1)).reshape(len(raw), seq_max_len, features.shape[-1])

        outputs = features_packed
        return outputs

    @staticmethod
    def load_data(parser, graph):
        raw = [n.word for n in graph.nodes[:]]

        return {'raw': raw}
