# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from data_loader import load_data
from train_eval import experiment
from plotting import save_accuracies_plot

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        train_loader, test_loader = load_data(SentimentDatasetBOW, batch_size=32)
        start_time = time.time()
        print(NN3BOW(input_size=512, hidden_size=100))
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        training_accuracies = {
                'NN2BOW': nn2_train_accuracy,
                'NN3BOW': nn3_train_accuracy
                }
        testing_accuracies = {
                'NN2BOW': nn2_test_accuracy,
                'NN3BOW': nn3_test_accuracy
                }
        save_accuracies_plot(training_accuracies, testing_accuracies)

    elif args.model == "DAN":
        #TODO:  Train and evaluate your DAN
        print("DAN model not implemented yet")

    
    elif args.model == "SUBWORDDAN":
        #TODO:  Train and evaluate your DAN
        print("SUBWORDDAN model not implemented yet")

    else:
        print("Model type not recognized.\nUse one of the following model types: BOW, DAN, SUBWORDDAN")

if __name__ == "__main__":
    main()
