import time
from data_loader import load_data_BOW, load_data_DAN
from tokenizer.BPETokenizer import BPETokenizer
from train_eval import experiment, objective
from plotting import save_accuracies_plot
import torch
from BOWmodels import NN2BOW, NN3BOW
from DANmodels import NN2DAN, OptimalDAN
import optuna


def optuna_study():
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
            study_name="DAN_hyperparameter_tuning",
            storage="sqlite:///DAN_hyperparameter_tuning.db",
            load_if_exists=True,
            direction="maximize",
            pruner=pruner)
    study.optimize(objective, n_trials=100)

def SUBWORDDAN_experiment(device:torch.device, visual=False):
    tokenizer = BPETokenizer(vocab_size=10000)
    train_loader, test_loader = load_data_DAN(batch_size=256, use_pretrained=False, embed_dims=50, tokenizer=tokenizer)
    basic_danmodel = NN2DAN(train_loader.get_embedding_layer(frozen=False), hidden_size=100,)
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, basic_danmodel, train_loader, test_loader)
    optimal_danmodel = OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False))
    optdan_train_accuracy, optdan_test_accuracy = experiment(device, optimal_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)

    training_accuracies = {
            'NN2BPEDAN': nn2_train_accuracy,
            'OPTBPEDAN': optdan_train_accuracy 
            }
    testing_accuracies = {
            'NN2BPEDAN': nn2_test_accuracy,
            'OPTBPEDAN': optdan_test_accuracy
            }
    save_accuracies_plot(training_accuracies, testing_accuracies)

def RANDOMDAN_experiment(device:torch.device):
    train_loader, test_loader = load_data_DAN(batch_size=256, use_pretrained=False, embed_dims=300)
    basic_danmodel = NN2DAN(train_loader.get_embedding_layer(frozen=False), hidden_size=100,)

    nn2_train_accuracy, nn2_test_accuracy = experiment(device, basic_danmodel, train_loader, test_loader)
    optimal_danmodel = OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False))
    optdan_train_accuracy, optdan_test_accuracy = experiment(device, optimal_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)

def DAN_experiment(device:torch.device, embed_dims:int=300):
    train_loader, test_loader = load_data_DAN(batch_size=256, embed_dims=embed_dims)
    basic_danmodel = NN2DAN(train_loader.get_embedding_layer(frozen=False), hidden_size=100,)

    print('\n2 layers:')
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, basic_danmodel, train_loader, test_loader)

    optimal_danmodel = OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False))
    optdan_train_accuracy, optdan_test_accuracy = experiment(device, optimal_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)

def BOW_experiment(device:torch.device):
    # Tests the accuracy of the two-layer and three-layer BOW models
    train_loader, test_loader = load_data_BOW(batch_size=32)

    # Train and evaluate NN2
    start_time = time.time()
    print('\n2 layers:')
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

    # Train and evaluate NN3
    print('\n3 layers:')
    nn3_train_accuracy, nn3_test_accuracy = experiment(device, NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)
    end_time = time.time()
