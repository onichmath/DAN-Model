# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from Documents.College.CSE256.hw1.CSE256_PA1_FA24.experiments import DAN_experiment
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from experiments import BOW_experiment, BOW_experiment_optuna

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--optuna', type=bool, required=False, default=False, help='Use Optuna for hyperparameter tuning')

    # Parse the command-line arguments
    args = parser.parse_args()
    # Check if the model type is "BOW"
    if args.optuna:
        BOW_experiment_optuna(device)
        print("Optuna hyperparameter tuning not implemented")
        exit()

    if args.model == "BOW":
        BOW_experiment(device)
    elif args.model == "DAN":
        #TODO:  Train and evaluate your DAN
        DAN_experiment(device)
        print("DAN model not implemented yet")
    
    elif args.model == "SUBWORDDAN":
        #TODO:  Train and evaluate your DAN
        print("SUBWORDDAN model not implemented yet")

    else:
        print("Model type not recognized.\nUse one of the following model types: BOW, DAN, SUBWORDDAN")

if __name__ == "__main__":
    main()
