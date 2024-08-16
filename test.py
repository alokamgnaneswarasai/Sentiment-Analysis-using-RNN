import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from preprocessing import load_data
from model import RNNClassifier
from dataloader import get_dataloader
from eval import evaluate
from dataloader import TextDataset

# Hyperparameters
max_seq_length = 50
batch_size = 32
hidden_dim = 128
output_dim = 5


# Load the preprocessed data
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
X_test, y_test = load_data('data.csv', max_seq_length)

# Create the dataloader

test_dataloader = get_dataloader(X_test, y_test, batch_size=batch_size)

# Load the model
model = RNNClassifier(300, 128, 5).to(device)
model.load_state_dict(torch.load('model.pth'))

# Evaluate the model
criterion = nn.CrossEntropyLoss()

print("Evaluating the model")
accuracy, _ = evaluate(model, test_dataloader, criterion, device)
print(f'Accuracy: {accuracy}')


