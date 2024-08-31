import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from preprocessing import load_test_data
from model import RNNClassifier, GRUClassifier, Bi_GRUClassifier , CNNClassifier
from dataloader import get_dataloader
from eval import evaluate
from dataloader import TextDataset
import argparse

max_seq_length = 100 # use 50 for electronics dataset and 10 for SST2 dataset
hidden_dim = 100
output_dim = 5 
input_dim = 300

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RNN',choices=['RNN','GRU','BI-GRU','CNN'], help='Model to use for training')
args = parser.parse_args()

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

X_test = load_test_data('electronics_validation.csv', max_seq_length)


if args.model == 'RNN':
    model = RNNClassifier(input_dim, hidden_dim, output_dim).to(device)
    
elif args.model == 'BI-GRU':
    model = Bi_GRUClassifier(input_dim, hidden_dim, output_dim).to(device)
   
elif args.model == 'GRU':
    model = GRUClassifier(input_dim, hidden_dim, output_dim).to(device)
    
else:
    model = CNNClassifier(input_dim, output_dim, num_filters=100, filter_sizes=[3,4,5]).to(device)
    
model_path = f'models/electronics/{args.model}_model.pth'
    
print(f'Loading model from {model_path}')
print(f'model selected: {args.model}')

model.load_state_dict(torch.load(model_path))


def predict(model,X_test):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_test)):
            X = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item()+1) # add 1 to get the ratings in the range 1-5
    return predictions

predictions = predict(model,X_test)
df = pd.DataFrame(predictions)
df.to_excel('ratings.xlsx',index=False,header=False)
print("Predictions saved to ratings.xlsx")



