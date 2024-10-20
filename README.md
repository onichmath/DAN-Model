# Deep Averaging Networks 
Deep Averaging Networks (DAN) are a type of vector space neural network designed for text classification. DAN models average the word embeddings of the words in a sentence to create fixed-length sentence embeddings. These sentence embeddings are passed through one or more feedforward layers to learn nonlinear patterns before being passed to a softmax layer for class predictions. 
[DAN Paper](https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)

### Installation

#### Clone the Repository
```
git clone git@github.com:onichmath/DAN-Model.git
```
#### Change Directory
```
cd DAN-Model
```
#### Create Virtual Environment
```
python -m venv venv
```
#### Activate Virtual Environment
```
source venv/bin/activate
```
#### Install packages from requirements.txt
```
pip install -r requirements.txt
```
#### Run Model
```
python main.py --model MODEL_NAME
```

### Usage
#### DAN Model
- The DAN model runs with pre-trained GloVe embeddings.
```
python main.py --model DAN
```
#### Random DAN Model
- The Random DAN model is a DAN model with randomly initialized word embeddings. 
```
python main.py --model RANDOMDAN
```
#### Subword DAN Model
- Note that the Subword DAN models uses BPE embeddings trained on the train dataset.
```
python main.py --model SUBWORDDAN
```
#### View Previous Studies
- You can view previous Optuna studies stored in the database.
```
optuna-dashboard sqlite:///DAN_hyperparameter_tuning.db
```
#### Optuna Study
- The Optuna study is used to find the best hyperparameters for the DAN model.
```
python main.py --optuna True
```
