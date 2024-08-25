import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from preprocessing import load_data
from model import RNNClassifier, GRUClassifier, Bi_GRUClassifier
from dataloader import get_dataloader
from eval import evaluate
from dataloader import TextDataset
import argparse

# Hyperparameters
max_seq_length = 20 # use 50 for electronics dataset and 10 for SST2 dataset
batch_size = 32
hidden_dim = 128
output_dim = 2 # 2 for SST2 dataset and 5 for electronics dataset
input_dim = 300

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RNN',choices=['RNN','GRU','BI-GRU'], help='Model to use for training')
args = parser.parse_args()



# Load the preprocessed data
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# X_test, y_test = load_data('data.csv', max_seq_length)
# X_test,y_test = load_data('electronics_validation.csv', 200)
X_test,y_test = load_data('data/SST2/validation.csv', max_seq_length, label_shifting=False)

# Create the dataloader

test_dataloader = get_dataloader(X_test, y_test, batch_size=batch_size)

# Load the model

if args.model == 'RNN':
    model = RNNClassifier(input_dim, hidden_dim, output_dim).to(device)
    
elif args.model == 'BI-GRU':
    model = Bi_GRUClassifier(input_dim, hidden_dim, output_dim).to(device)
   
elif args.model == 'GRU':
    model = GRUClassifier(input_dim, hidden_dim, output_dim).to(device)
    
model_path = f'models/{args.model}_model.pth'
    
print(f'Loading model from {model_path}')
print(f'model selected: {args.model}')

model.load_state_dict(torch.load(model_path))
criterion = nn.CrossEntropyLoss()

print("Evaluating the model")
accuracy, _ ,f1_score= evaluate(model, test_dataloader, criterion, device)
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1_score}')


