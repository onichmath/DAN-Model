from collections import defaultdict
import os
import re
import json
from typing import Counter
import time
from sentiment_data import read_sentiment_examples

class BPETokenizer():
    def __init__(self, vocab_size):
        start = time.time()
        self.base_vocab_size = 256
        self.vocab_size = vocab_size
        self.merges = self.load_merges(self.vocab_size)
        self.vocab = self.build_vocab()
        end = time.time()
        print(f"Loaded BPE tokenizer with vocab size {vocab_size} in {end-start} seconds")

    def encode(self, sentence):
        # Encode sentence using BPE
        if "<s>" not in sentence[-1]:
            sentence[-1] = sentence[-1] + "</s>"
        sentence = " ".join(sentence)
        tokens = list(sentence.encode("utf-8"))

        while True:
            stats = self.get_stats(tokens)
            if not stats:
                break 
            # Use float inf as fallback when pair not in merges
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break # Nothing to merge
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        # Decode tokens using BPE Vocab
        tokens = b"".join([self.vocab[i] for i in ids])
        text = tokens.decode("utf-8", errors="replace")
        return text

    def build_vocab(self):
        # Build vocab from merges
        vocab = {i: bytes([i]) for i in range(self.base_vocab_size)}
        for (p0, p1), i in self.merges.items():
            vocab[i] = vocab[p0] + vocab[p1]
        return vocab

    def load_merges(self, vocab_size):
        # Load merges from file
        if os.path.exists(f"./tokenizer/bpe_{vocab_size}.json"):
            with open(f"./tokenizer/bpe_{vocab_size}.json", "r") as f:
                merges = json.load(f)
            merges = {tuple(map(int, key.split(','))): val for key, val in merges.items()}
            return merges
        else:
            raise ValueError(f"""
                             Could not find merges file for vocab size {vocab_size}.
                             Please train BPE tokenizer with `python main.py --model BPE`
                             """)

    @staticmethod
    def get_stats(ids) -> Counter:
        # Get frequency of pairs of ids
        counts = Counter(zip(ids, ids[1:]))
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

        base_vocab_size = 256

        merges = {}
        for vocab_size in sorted(vocab_sizes):
            print(f"Training BPE with vocab size {vocab_size}")
            stats = BPETokenizer.get_stats(ids)
            while len(merges) < vocab_size - base_vocab_size:
                if not stats:
                    break
                new_vocab_size = len(merges) + base_vocab_size
                top_pair = stats.most_common(1)[0][0]

                ids = BPETokenizer.merge(ids, top_pair, new_vocab_size)
                merges[','.join(map(str,top_pair))] = new_vocab_size

                stats = BPETokenizer.get_stats(ids)

            print(f"Vocab size: {vocab_size} ids: {len(ids)} merges: {len(merges)}")
            with open(f"./tokenizer/bpe_{vocab_size}.json", "w") as f:
                json.dump(merges, f)
