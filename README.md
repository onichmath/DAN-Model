# Deep Averaging Networks 
Deep Averaging Networks (DAN) are a type of vector space neural network designed for text classification. DAN models average the word embeddings of the words in a sentence to create fixed-length sentence embeddings. These sentence embeddings are passed through one or more feedforward layers to learn nonlinear patterns before being passed to a softmax layer for class predictions. 
[DAN Paper](https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)

## Usage 
### DAN Model
```
python main.py --model DAN
```
### Subword DAN Model
```
python main.py --model SUBWORDDAN
```