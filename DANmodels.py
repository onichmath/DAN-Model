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
        # Should -1 UNK indices be changed to 1?
        # Get the sentence at the given index
        sentence = self.sentences[idx]
        word_indices = torch.tensor([
            self.embeddings.word_indexer.index_of(word) for word in sentence.split()
        ], dtype=torch.int)

        word_indices = torch.where(word_indices == -1, torch.tensor(1, dtype=torch.int), word_indices)

        label = self.labels[idx]
        print(word_indices.dtype)

        return word_indices, label

class NN2DAN(nn.Module):
    def __init__(self, word_embedding_layer, hidden_size):
        super().__init__()
        self.embedding = word_embedding_layer
        self.fc1 = nn.Linear(self.embedding.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
