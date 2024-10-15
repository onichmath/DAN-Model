import torch
from torch import nn
from torch.nn.modules.activation import F
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        self.embeddings = word_embeddings
        
        # Convert labels to PyTorch tensors
        # self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return sete of word indices and labels for the given index
        word_indices = [self.embeddings.word_indexer.index_of(word) for word in self.sentences[idx]]
        return torch.tensor(word_indices), self.labels[idx]

class NN2DAN(nn.Module):
    def __init__(self, input_size, hidden_size, word_embedding_layer):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
