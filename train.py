import torch
import torch.nn as nn
from torch.optim import Adam
from dataloader import get_dataloader
from model import RNNClassifier, GRUClassifier
from eval import evaluate
from preprocessing import load_data
import argparse

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def train(model,dataloader,criterion,optimizer,device,num_epochs=10):
    
        
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for text, labels in dataloader:
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(text)
            
            loss = criterion(outputs, labels)
            
           
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {loss.item()}')
        
        accuracy, loss = evaluate(model, dataloader,criterion,device)
        print(f'Accuracy: {accuracy} Loss: {loss}')
        # print(f'Report: {report}')
        
    # torch.save(model.state_dict(), 'model.pth')
    # torch.save(model.state_dict(), f'{args.model}_model.pth')
    # Instead save in models folder
    torch.save(model.state_dict(), f'models/{args.model}_model.pth')
    
    print("Model trained and saved successfully in path ",f'models/{args.model}_model.pth')
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RNN',choices=['RNN','GRU'], help='Model to use for training')
    args = parser.parse_args()
    
    
    #Hyperparameters
    max_seq_length = 20 # use 50 for electronics dataset and 10 for SST2 dataset
    input_dim = 300
    hidden_dim = 128
    output_dim = 2
    num_epochs = 50
    learning_rate = 0.0025 # use 0.001 for electronics dataset and 0.0025 for SST2 dataset
    batch_size = 1024 # use 32 for electronics dataset and 1024 for SST2 dataset
    
    #Load the preprocessed data
    # X,y = load_data('data.csv', max_seq_length)
    
    X,y = load_data('data/SST2/train.csv', max_seq_length,label_shifting=False)
    
    
    # print(X[0],y[0])
    

    
    # Create the dataloader
    dataloader = get_dataloader(X, y, batch_size=batch_size)
    
    
    print("Creating the model")
    
    # Create the model
    
    if args.model == 'RNN':
        
        model = RNNClassifier(input_dim, hidden_dim, output_dim).to(device)
    else:
        model = GRUClassifier(input_dim, hidden_dim, output_dim).to(device)
        
    
   
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print("Training the model")
    train(model,dataloader,criterion,optimizer,device,num_epochs=num_epochs)
    
    
    # Save the model
    
    