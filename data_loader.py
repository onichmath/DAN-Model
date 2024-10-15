import time 
from torch.utils.data import DataLoader, Dataset
from BOWmodels import SentimentDatasetBOW
from DANmodels import SentimentDatasetDAN

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

def load_data_DAN(batch_size=32, glove_dims=300):
    start_time = time.time()
    if glove_dims == 50:
        glove_file = "./data/glove.6B.50d-relativized.txt"
    elif glove_dims == 300:
        glove_file = "./data/glove.6B.300d-relativized.txt"
    else:
        raise ValueError("Invalid glove dimension")

    train_data = SentimentDatasetDAN("data/train.txt", glove_file)
    test_data = SentimentDatasetDAN("data/dev.txt", glove_file)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"DAN: Data loaded in : {elapsed_time} seconds")
    return train_loader, test_loader
