from time import time
import torch
from torch import nn
from torch.nn.modules.activation import F
from utils.sentiment_data import WordEmbeddings, read_sentiment_examples
from torch.utils.data import Dataset
import numpy as np


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings=None, embed_dim=300, pretrained=False, tokenizer=None, train=True):
        # Pass in vectorizer? tokenizer
        # If pretraining ..;
        # Read the sentiment examples from the input file
        self.embed_dim = embed_dim

        self.examples = read_sentiment_examples(infile)
        # self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.sentences = [ex.words for ex in self.examples]
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)
        self.train = "train" if train else "test"

        if tokenizer:
            print(f"Tokenizing sentences in {self.train} dataset")
            start = time()
            if pretrained:
                raise ValueError("Cannot pass in tokenizer for pretrained model")
            self.tokenizer = tokenizer
            self.sentences = [self.tokenizer.encode(sent) for sent in self.sentences]
            end = time()
            print(f"Tokenization took {end - start} seconds")
        else:
            self.tokenizer = None

        if not train:
            # If not training, set word embeddings
            # Assumes word embeddings made in train set
            if not word_embeddings:
                raise ValueError("Need to pass in word embeddings for test set")
            self.embeddings = word_embeddings
        if train:
            if pretrained:
                # Load pretrained model
                print(f"Loading pretrained embeddings in {self.train} dataset")
                self.embeddings = self._load_pretrained_embeddings()
            else:
                if not train and not word_embeddings:
                    raise ValueError("Need to pass in word embeddings for non-pretrained model")
                print(f"Randomly initializing {self.train} embeddings")
                self.embeddings = self._randomly_initialize_embeddings()
        print(f"Precomputing padded token indices for {self.train} dataset")
        self.token_indices = self._precompute_padded_token_indices()

    def get_embedding_layer(self, frozen=False):
        """
        Get the embedding layer for the dataset
        """
        return self.embeddings.get_initialized_embedding_layer(frozen=frozen)

    def _randomly_initialize_embeddings(self):
        """
        Randomly initializes embeddings for the tokens in the dataset
        """
        all_words = [word for sent in self.sentences for word in sent]
        unique_words = np.unique(all_words)
        return WordEmbeddings.get_randomly_initialized_embeddings(unique_words, self.embed_dim)

    def _load_pretrained_embeddings(self):
        """
        Load pretrained glove embeddings from file
        """
        if self.embed_dim != 50 and self.embed_dim != 300:
            raise ValueError("Invalid glove dimension")
        glove_file = f"./data/glove.6B.{self.embed_dim}d-relativized.txt"
        return WordEmbeddings.read_word_embeddings(glove_file)

    def _precompute_padded_token_indices(self):
        """
        Precompute the word indices for each sentence and pad each sentence to the length of the longest sentence
        Sets padding indices to 0, as done in read_word_embeddings
        """
        if not self.embeddings:
            raise ValueError("Need word embeddings to precompute word indices")
        max_len = max(len(sent) for sent in self.sentences)
        token_indices = []
        for sentence in self.sentences:
            indices = [self.embeddings.word_indexer.index_of(token) for token in sentence]
            indices += [0] * (max_len - len(indices))
            token_indices.append(indices)

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

    def mean_ignore_padding(self, tensor: torch.Tensor):
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
    def __init__(self, word_embedding_layer, hidden_size=300, dropout_prob=0.3, n_layers=2):
        super().__init__()
        self.embedding = word_embedding_layer
        self.embedfc = nn.Linear(self.embedding.embedding_dim, hidden_size)
        self.sequence = nn.Sequential()
        for i in range(n_layers):
            self.sequence.add_module(f"fc{i}", nn.Linear(hidden_size, hidden_size))
            self.sequence.add_module(f"relu{i}", nn.ReLU())
            self.sequence.add_module(f"dropout{i}", nn.Dropout(dropout_prob))
        self.outfc = nn.Linear(hidden_size, 2)

        self.dropout = nn.Dropout(dropout_prob)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.mean_ignore_padding(x)

        x = self.embedfc(x)
        x = self.dropout(x)
        x = self.sequence(x)
        x = self.outfc(x)
        x = self.log_softmax(x)
        return x
