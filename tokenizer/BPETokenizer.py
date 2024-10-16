import os

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
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    @staticmethod 
    def train_bpe(vocab_sizes=[1000, 5000, 10000, 20000, 50000, 100000]):
        # Read in labels, sentences
        text = read_sentiment_examples("./data/train.txt")
        # Convert to list of words
        text = [ex.words for ex in text]
        # Conver matrix of words to list of integers corresponding to words
        all_words = " ".join([" ".join(ex) for ex in text])
        tokens = list(map(int, all_words.encode("utf-8")))
        print(len(tokens))
        stats = BPETokenizer.get_stats(tokens)
        top_pair = max(stats, key=stats.get)
        tokens = BPETokenizer.merge(tokens, top_pair, 256)
        print(len(tokens))
        # print(stats)
        # stats = sorted(((v,k) for k,v in stats.items()), reverse=True)
        # print(stats)


        pass
