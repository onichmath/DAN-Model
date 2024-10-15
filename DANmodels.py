import torch
from torch import nn
from torch.nn.modules.activation import F
from torch.nn.utils.rnn import pad_sequence
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
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        self.word_indices = self._precompute_padded_word_indices()

    def _precompute_padded_word_indices(self):
        """
        Precompute the word indices for each sentence and pad each sentence to the length of the longest sentence
        Sets padding indices to 0, as done in read_word_embeddings
        """
        # UNK indices not set here, asssumed to be handled by embedding layer
        max_len = max(len(sent.split()) for sent in self.sentences)
        word_indices = []
        for sentence in self.sentences:
            indices = [self.embeddings.word_indexer.index_of(word) for word in sentence.split()]
            indices += [0] * (max_len - len(indices))
            word_indices.append(indices)
        word_indices = torch.tensor(word_indices, dtype=torch.int)
        word_indices = torch.where(word_indices == -1, torch.tensor(1, dtype=torch.int), word_indices)

        return word_indices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return sete of word indices and labels for the given index
        return self.word_indices[idx], self.labels[idx]

class DANModel(nn.Module):
    def __init__(self):
        super().__init__()

    def mean_ignore_padding(self, tensor):
        # Calculate the mean while ignoring 0s (padding tokens)
        mask = tensor != 0
        sum_values = torch.sum(tensor * mask, dim=1)
        nonzero_counts = torch.sum(mask, dim=1)
        nonzero_counts[nonzero_counts == 0] = 1
        return sum_values / nonzero_counts

class NN2DAN(DANModel):
    def __init__(self, word_embedding_layer, hidden_size):
        super().__init__()
        self.embedding = word_embedding_layer
        self.fc1 = nn.Linear(self.embedding.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mean_ignore_padding(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

