from DANmodels import NN2DAN, OptimalDAN
from data_loader import load_data_DAN
import torch
from torch import nn
import optuna
from tokenizer.BPETokenizer import BPETokenizer

def objective(trial):
    # An optuna trial objective 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)
    embed_dims = trial.suggest_int("embed_dims", 50, 500, step=50)
    use_pretrained = trial.suggest_categorical("use_pretrained", [True, False])
    tokenizer = trial.suggest_categorical("tokenizer", [None, 
                                                        BPETokenizer(vocab_size=1000),
                                                        BPETokenizer(vocab_size=5000),
                                                        BPETokenizer(vocab_size=10000),
                                                        BPETokenizer(vocab_size=20000),
                                                        BPETokenizer(vocab_size=50000),
                                                        BPETokenizer(vocab_size=100000)])
    hidden_size = trial.suggest_int("hidden_size", 50, 500, step=50)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    train_loader, test_loader = load_data_DAN(batch_size=batch_size, embed_dims=embed_dims, use_pretrained=use_pretrained, tokenizer=tokenizer)

    model = trial.suggest_categorical("model", [
        NN2DAN(train_loader.get_embedding_layer(frozen=False), hidden_size=hidden_size),
        OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False), hidden_size=hidden_size)])
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
            trial.report(test_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return max(all_test_accuracy) if all_test_accuracy else 0.0


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer: torch.optim.Optimizer, device):
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
