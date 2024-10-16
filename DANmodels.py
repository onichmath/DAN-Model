import torch
from torch import nn
from torch.nn.modules.activation import F
from sentiment_data import WordEmbeddings, read_sentiment_examples
from torch.utils.data import Dataset
import numpy as np


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings=None, vocab_size=15000, embed_dim=300, pretrained=True, tokenizer=None, train=True):
        # Pass in vectorizer? tokenizer
        # If pretraining ..;
        # Read the sentiment examples from the input file
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.examples = read_sentiment_examples(infile)
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        if tokenizer:
            if pretrained:
                raise ValueError("Cannot pass in tokenizer for pretrained model")
            self.tokenizer = tokenizer
            self.sentences = [self.tokenizer.tokenize(sent) for sent in self.sentences]

        if not train:
            # If not training, set word embeddings
            # Assumes word embeddings made in train set
            if not word_embeddings:
                raise ValueError("Need to pass in word embeddings for test set")
            self.embeddings = word_embeddings
            self.token_indices = self._precompute_padded_token_indices()
        if train:
            if pretrained:
                # Load pretrained model
                self.embeddings = self._load_pretrained_embeddings()
                self.token_indices = self._precompute_padded_token_indices()
            else:
                if not train and not word_embeddings:
                    raise ValueError("Need to pass in word embeddings for non-pretrained model")
                self.embeddings = self._randomly_initialize_embeddings()
                self.token_indices = self._precompute_padded_token_indices()

    def get_embedding_layer(self, frozen=False):
        return self.embeddings.get_initialized_embedding_layer(frozen=frozen)

    def _randomly_initialize_embeddings(self):
        return WordEmbeddings.get_randomly_initialized_embeddings(np.unique(self.sentences).flatten(), self.embed_dim)

    def _load_pretrained_embeddings(self):
        if self.embed_dim != 50 and self.embed_dim != 300:
            raise ValueError("Invalid glove dimension")
        glove_file = f"./data/glove.6B.{self.embed_dim}d-relativized.txt"
        return WordEmbeddings.read_word_embeddings(glove_file)

    def _precompute_padded_token_indices(self):
        """
        Precompute the word indices for each sentence and pad each sentence to the length of the longest sentence
        Sets padding indices to 0, as done in read_word_embeddings
        """
        # UNK indices not set here, asssumed to be handled by embedding layer
        if not self.embeddings:
            raise ValueError("Need word embeddings to precompute word indices")
        max_len = max(len(sent.split()) for sent in self.sentences)
        token_indices = []
        for sentence in self.sentences:
            indices = [self.embeddings.word_indexer.index_of(token) for token in sentence.split()]
            indices += [0] * (max_len - len(indices))
            token_indices.append(indices)

        print(token_indices)
        token_indices = torch.tensor(token_indices, dtype=torch.int)
        token_indices = torch.where(token_indices == -1, torch.tensor(1, dtype=torch.int), token_indices)

        return token_indices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return sete of word indices and labels for the given index
        return self.token_indices[idx], self.labels[idx]

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

class OptimalDAN(DANModel):
    # Implements the best performing DAN model in the DAN paper:  https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
    def __init__(self, word_embedding_layer, hidden_size=300, dropout_prob=0.3):
        super().__init__()
        self.embedding = word_embedding_layer
        self.fc1 = nn.Linear(self.embedding.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)

        self.dropout = nn.Dropout(dropout_prob)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.mean_ignore_padding(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.log_softmax(x)
        return x
