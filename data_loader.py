import time
from typing import cast
import torch 
from torch.utils.data import DataLoader, Dataset
from BOWmodels import SentimentDatasetBOW
from DANmodels import SentimentDatasetDAN
from torch.nn.utils.rnn import pad_sequence

class DANDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_embedding_layer(self, frozen=False):
        dataset = cast(SentimentDatasetDAN, self.dataset)
        return dataset.get_embedding_layer(frozen=frozen)

def load_data_BOW(batch_size=32):
    # Load dataset using a given data class
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"BOW: Data loaded in : {elapsed_time} seconds")
    return train_loader, test_loader

def padded_collate_fn(batch):
    # Pads a batch of sequences to the same length
    # Uses 0 as padding token, as done in read_word_embeddings
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts,  labels

def load_data_DAN(batch_size=32, embed_dims=300, use_pretrained=True, vocab_size=15000, tokenizer=None):
    start_time = time.time()

    train_data = SentimentDatasetDAN("data/train.txt", embed_dim=embed_dims, pretrained=use_pretrained, train=True, vocab_size=vocab_size, tokenizer=tokenizer)
    test_data = SentimentDatasetDAN("data/dev.txt", word_embeddings=train_data.embeddings, train=False, vocab_size=vocab_size, tokenizer=tokenizer)

    train_loader = DANDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DANDataLoader(test_data, batch_size=batch_size, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"DAN: Data loaded in : {elapsed_time} seconds")
    return train_loader, test_loader
