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
    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod 
    def train_bpe(vocab_sizes=[1000, 5000, 10000, 20000, 50000, 100000]):
        # Read in labels, sentences
        text = read_sentiment_examples("./data/train.txt")
        # Convert to list of words
        text = [ex.words for ex in text]
        # Conver matrix of words to list of integers corresponding to words
        all_words = " ".join([" ".join(ex) for ex in text])
        print(len(all_words))
        tokens = list(map(int, all_words.encode("utf-8")))
        print(len(tokens))
        print(max(tokens))
        stats = BPETokenizer.get_stats(tokens)
        print(stats)
        stats = sorted(((v,k) for k,v in stats.items()), reverse=True)
        print(stats)


        pass
