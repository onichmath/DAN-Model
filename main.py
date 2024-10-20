# models.py
import torch
import argparse
from training.experiments import BOW_experiment, DAN_experiment, RANDOMDAN_experiment, SUBWORDDAN_experiment, optuna_study
from tokenizer.BPETokenizer import BPETokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=False, default="DAN", help='Model type to train (e.g., BOW)')
    parser.add_argument('--optuna', type=bool, required=False, default=False, help='Use Optuna for hyperparameter tuning')

    # Parse the command-line arguments
    args = parser.parse_args()
    # Check if the model type is "BOW"
    if args.optuna:
        optuna_study()
        exit()
    if args.model == "BOW":
        BOW_experiment(device)
    elif args.model == "DAN":
        DAN_experiment(device)
    elif args.model == "RANDOMDAN":
        RANDOMDAN_experiment(device)
    elif args.model == "SUBWORDDAN":
        SUBWORDDAN_experiment(device)
    elif args.model == "BPE":
        # Train BPE tokenizer
        BPETokenizer.train_bpe()
    else:
        print("Model type not recognized.\nUse one of the following model types: BOW, DAN, SUBWORDDAN")

if __name__ == "__main__":
    main()
