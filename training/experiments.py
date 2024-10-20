from training.data_loader import load_data_BOW, load_data_DAN
from tokenizer.BPETokenizer import BPETokenizer
from training.train_eval import experiment, objective
from utils.plotting import save_accuracies_plot
import torch
from models.BOWmodels import NN2BOW, NN3BOW
from models.DANmodels import NN2DAN, OptimalDAN
import optuna


def optuna_study():
    """
    Conducts an optuna study to find the optimal hyperparameters for the OptimalDAN model
    """
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
            study_name="Optimal DAN",
            storage="sqlite:///DAN_hyperparameter_tuning.db",
            load_if_exists=True,
            direction="maximize",
            pruner=pruner)
    study.optimize(objective, n_trials=1000)

def SUBWORDDAN_experiment(device:torch.device):
    """
    Tests the accuracy of the two-layer and three-layer DAN models with byte pair tokenized embeddings
    """
    tokenizer = BPETokenizer(vocab_size=10000)
    train_loader, test_loader = load_data_DAN(batch_size=256, use_pretrained=False, embed_dims=300, tokenizer=tokenizer)
    basic_danmodel = NN2DAN(train_loader.get_embedding_layer(frozen=False), hidden_size=100,)
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, basic_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)
    optimal_danmodel = OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False))
    optdan_train_accuracy, optdan_test_accuracy = experiment(device, optimal_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)

    training_accuracies = {
            "NN2DAN": nn2_train_accuracy,
            "OptimalDAN": optdan_train_accuracy
            }
    testing_accuracies = {
            "NN2DAN": nn2_test_accuracy,
            "OptimalDAN": optdan_test_accuracy
            }
    save_accuracies_plot(training_accuracies, testing_accuracies)

def RANDOMDAN_experiment(device:torch.device):
    """
    Tests the accuracy of the two-layer and three-layer DAN models with randomly initialized embeddings
    """
    train_loader, test_loader = load_data_DAN(batch_size=256, use_pretrained=False, embed_dims=300)
    basic_danmodel = NN2DAN(train_loader.get_embedding_layer(frozen=False), hidden_size=100,)

    nn2_train_accuracy, nn2_test_accuracy = experiment(device, basic_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)
    optimal_danmodel = OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False))
    optdan_train_accuracy, optdan_test_accuracy = experiment(device, optimal_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)
    training_accuracies = {
            "NN2DAN": nn2_train_accuracy,
            "OptimalDAN": optdan_train_accuracy
            }
    testing_accuracies = {
            "NN2DAN": nn2_test_accuracy,
            "OptimalDAN": optdan_test_accuracy
            }
    save_accuracies_plot(training_accuracies, testing_accuracies)

def DAN_experiment(device:torch.device, embed_dims:int=300):
    """
    Tests the accuracy of the two-layer and three-layer DAN models with pretrained embeddings
    """
    train_loader, test_loader = load_data_DAN(batch_size=256, embed_dims=embed_dims, use_pretrained=True)
    basic_danmodel = NN2DAN(train_loader.get_embedding_layer(frozen=False), hidden_size=100,)

    print('\n2 layers:')
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, basic_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)

    optimal_danmodel = OptimalDAN(word_embedding_layer=train_loader.get_embedding_layer(frozen=False))
    optdan_train_accuracy, optdan_test_accuracy = experiment(device, optimal_danmodel, train_loader, test_loader, learning_rate=0.0001, weight_decay=1e-5)

    training_accuracies = {
            "NN2DAN": nn2_train_accuracy,
            "OptimalDAN": optdan_train_accuracy
            }
    testing_accuracies = {
            "NN2DAN": nn2_test_accuracy,
            "OptimalDAN": optdan_test_accuracy
            }
    save_accuracies_plot(training_accuracies, testing_accuracies)


def BOW_experiment(device:torch.device):
    # Tests the accuracy of the two-layer and three-layer BOW models
    train_loader, test_loader = load_data_BOW(batch_size=32)

    # Train and evaluate NN2
    nn2_train_accuracy, nn2_test_accuracy = experiment(device, NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

    # Train and evaluate NN3
    nn3_train_accuracy, nn3_test_accuracy = experiment(device, NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)
