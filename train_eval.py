from DANmodels import OptimalDAN
from data_loader import load_data_DAN
import torch
from torch import nn
from tokenizer.BPETokenizer import BPETokenizer

def objective(trial):
    """
    An optuna trial objective 
    """
    # An optuna trial objective 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = trial.suggest_int("Batch Size", 32, 512, step=32)
    embed_dims = trial.suggest_int("Embedding Dimensions", 32, 512, step=32)
    hidden_size = trial.suggest_int("Hidden Size", 32, 512, step=32)
    use_pretrained = False

    tokenizer_size = trial.suggest_categorical("Tokenizer Vocab Size", [0, 1000, 5000, 10000, 20000, 50000])
    tokenizer = None
    if tokenizer_size > 0:
        tokenizer = BPETokenizer(vocab_size=tokenizer_size)

    learning_rate = trial.suggest_float("Learning Rate", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("Weight Decay", 1e-7, 1e-3, log=True)

    train_loader, test_loader = load_data_DAN(batch_size=batch_size, embed_dims=embed_dims, use_pretrained=use_pretrained, tokenizer=tokenizer)

    n_layers = trial.suggest_int("Hidden Layers", 0, 4)
    dropout_rate = trial.suggest_float("Dropout Rate", 0.0, 0.5)
    model = OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False), hidden_size=hidden_size, dropout_prob=dropout_rate, n_layers=n_layers)

    model = model.to(device)
    loss_fn = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    all_train_accuracy = []
    all_test_accuracy = []
    print(trial.params)
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer, device)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
        trial.report(test_accuracy, epoch)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()
    return max(all_test_accuracy) if all_test_accuracy else 0.0


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer: torch.optim.Optimizer, device):
    """
    Train the model for one epoch
    """
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer, device):
    """
    Evaluate the model on the dev set
    """
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(device:torch.device, model:nn.Module, train_loader, test_loader, learning_rate=0.0001, weight_decay=0.0):
    """
    Train and evaluate the model for multiple epochs
    """
    model = model.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer, device)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy
