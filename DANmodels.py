import torch
from torch import nn
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, glove_file, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        # Read in word embeddings
        self.embeddings = read_word_embeddings(glove_file)
        
        # Convert labels to PyTorch tensors
        # self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return sete of word indices and labels for the given index
        word_indices = [self.embeddings.word_indexer.index_of(word) for word in self.sentences[idx]]
        return torch.tensor(word_indices), self.labels[idx]

    def get_embedding_layer(self, frozen=True):
        return self.embeddings.get_initialized_embedding_layer(frozen=frozen)


