import time 
from torch.utils.data import DataLoader

def load_data(data_class, batch_size=32):
    # Load dataset
    start_time = time.time()

    train_data = data_class("data/train.txt")
    dev_data = data_class("data/dev.txt")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{data_class}: Data loaded in : {elapsed_time} seconds")
    return train_loader, test_loader
