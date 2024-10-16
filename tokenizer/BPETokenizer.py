from collections import defaultdict
import os
import re

from numpy._core.defchararray import join

from sentiment_data import read_sentiment_examples

class BPETokenizer():
    def __init__(self, vocab_len):
        # Read vocab file for given length if exists
        pass

    def encode(self, sentence):
        # Encode sentence using BPE
        pass

    def decode(self, tokens):
        # Decode tokens using BPE
        pass

    @staticmethod
    def get_stats(ids) -> dict:
        # Get frequency of pairs of ids
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod 
    def merge(ids, pair, idx):
        # In list of ids, replace all occurrences of pair with idx
        new_ids = []
        i = 0
        while i < len(ids):
            # If match, replace with idx and skip next
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    @staticmethod 
    def split_sentence(sentence):
        return [" ".join(word) + " </w>" for word in sentence] + ["</s>"]


    @staticmethod 
    def train_bpe(vocab_sizes=[1000, 5000, 10000, 20000, 50000, 100000]):
        # Read in labels, sentences
        text = read_sentiment_examples("./data/train.txt")
        print(f"Read in {len(text)} examples")
        # Get list of words
        text = [" ".join(sent.words) for sent in text]

        all_words = "</s> ".join(text)
        tokens = list(map(int, all_words.encode("utf-8")))
        ids = list(tokens)

        for vocab_size in sorted(vocab_sizes):
            print(f"Training BPE with vocab size {vocab_size}")
            num_merges = vocab_size - 256
            merges = {}

            for i in range(num_merges):
                stats = BPETokenizer.get_stats(ids)
                top_pair = max(stats, key=stats.get)
                if not top_pair in stats:
                    break
                idx = 256 + i
                ids = BPETokenizer.merge(ids, top_pair, idx)
                print(f"Pair: {top_pair} -> {idx} -> {len(ids)}")
                merges[top_pair] = idx

            print(f"Vocab size: {vocab_size} ids: {len(ids)}")

        print(len(ids))


        pass
