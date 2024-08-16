## Assignment1
- Implement BP (using any existing tool/platform not allowed)
- Think of the correct architecture for Palindrome
- Train a feedforward n/w for solving the 10-bit palindrome problem (input- bit strings of 1 and 0); there will be 1024 input strings labeled 1 (if the string is Palindrome) and 0 (non-P)
- Train and Test using 4-fold cross-validation
Measure Precision
- Find out what the hidden layer neurons are doing (VIMP)

### Main file
cs772_assignment1_final.ipynb
- run on google colab
- most of things are expain in this file


BP is implemented withouth using any existing tool/platform.

for other than main task library used are:
- import numpy as np
- import matplotlib.pyplot as plt
- from tqdm import tqdm
- import pandas as pd
- from sklearn.metrics import classification_report

Neural Network visualization and demo interface are added in the end of main file, can be directily runs (trained weights are used)

Cross validation results for each 4 fold (weight and bias are printed for each fold)

hidden layer neurons interpretaion
- page 6 of presentation slides
- can be understant from the weights shown in main file

### other version of implemetations
single_hidden_neuron_experiments

#### Experiment with single neuron in hidden layers

- cd single_hidden_neuron_experiments
- python single_hidden_palindrome.py


single_hidden_palindrome.py has the weights for single neuron in hidden layer in case of square activations