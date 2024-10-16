import os

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
    def train_bpe():
        text_file = "./data/train.txt"
        if not os.path.exists(text_file):
            print("Training file not found")
            return

        vocab_sizes = [1000, 3000, 5000, 10000, 20000, 50000]
        pass
