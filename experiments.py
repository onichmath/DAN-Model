import time
from data_loader import load_data_BOW, load_data_DAN
from sentiment_data import WordEmbeddings
from train_eval import experiment
from plotting import save_accuracies_plot
import torch
from BOWmodels import NN2BOW, NN3BOW
from DANmodels import NN2DAN, OptimalDAN
import optuna

def SUBWORDDAN_experiment(device:torch.device):
    pass

def DAN_experiment(device:torch.device, glove_dims:int=300):
    # device = torch.device("cpu")
    train_loader, test_loader, word_embeddings = load_data_DAN(batch_size=32, glove_dims=glove_dims)
    basic_danmodel = NN2DAN(word_embedding_layer=word_embeddings.get_initialized_embedding_layer(), hidden_size=100,)

    start_time = time.time()
    print('\n2 layers:')
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, basic_danmodel, train_loader, test_loader)
    # print(nn2_train_accuracy, nn2_test_accuracy)

    optimal_danmodel = OptimalDAN(word_embedding_layer=word_embeddings.get_initialized_embedding_layer())
    optdan_train_accuracy, optdan_test_accuracy = experiment(device, optimal_danmodel, train_loader, test_loader, learning_rate=0.001, weight_decay=1e-5)
    stop_time = time.time()

    training_accuracies = {
            'NN2DAN': nn2_train_accuracy,
            'OPTDAN': optdan_train_accuracy 
            }
    testing_accuracies = {
            'NN2DAN': nn2_test_accuracy,
            'OPTDAN': optdan_test_accuracy
            }
    save_accuracies_plot(training_accuracies, testing_accuracies)




def DAN_experiment_optuna(device:torch.device):
    pass


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

    training_accuracies = {
            'NN2BOW': nn2_train_accuracy,
            'NN3BOW': nn3_train_accuracy
            }
    testing_accuracies = {
            'NN2BOW': nn2_test_accuracy,
            'NN3BOW': nn3_test_accuracy
            }
    save_accuracies_plot(training_accuracies, testing_accuracies)
