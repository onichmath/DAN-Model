import time
from data_loader import load_data
from train_eval import experiment
from plotting import save_accuracies_plot
import torch
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
import optuna

def SUBWORDDAN_experiment(device:torch.device):
    pass

def DAN_experiment(device:torch.device):
    pass

def DAN_experiment_optuna(device:torch.device):
    pass

def BOW_experiment_optuna(device:torch.device):
    pass

def BOW_experiment(device:torch.device):
    # Tests the accuracy of the two-layer and three-layer BOW models
    train_loader, test_loader = load_data(SentimentDatasetBOW, batch_size=32)

    # Train and evaluate NN2
    start_time = time.time()
    print('\n2 layers:')
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

    # Train and evaluate NN3
    print('\n3 layers:')
    nn3_train_accuracy, nn3_test_accuracy = experiment(device, NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)
    end_time = time.time()

    training_accuracies = {
            'NN2BOW': nn2_train_accuracy,
            'NN3BOW': nn3_train_accuracy
            }
    testing_accuracies = {
            'NN2BOW': nn2_test_accuracy,
            'NN3BOW': nn3_test_accuracy
            }
    save_accuracies_plot(training_accuracies, testing_accuracies)
