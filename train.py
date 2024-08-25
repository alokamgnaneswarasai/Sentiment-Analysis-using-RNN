import torch
import torch.nn as nn
from torch.optim import Adam
from dataloader import get_dataloader
from model import RNNClassifier, GRUClassifier , Bi_GRUClassifier , CNNClassifier
from eval import evaluate
from preprocessing import load_data
import argparse
from sklearn.metrics import f1_score

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def train(model,train_dataloader,val_dataloader,criterion,optimizer,device,num_epochs=10):
    
        
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for text, labels in train_dataloader:
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {loss.item()}')
        train_accuracy, train_loss ,_= evaluate(model, train_dataloader,criterion,device)
        val_accuracy, val_loss ,_= evaluate(model, val_dataloader,criterion,device)
        print(f'Train Accuracy: {train_accuracy} Train Loss: {train_loss} Val Accuracy: {val_accuracy} Val Loss: {val_loss}')
 
    torch.save(model.state_dict(), f'models/{args.model}_model.pth')
    
    print("Model trained and saved successfully in path ",f'models/{args.model}_model.pth')
    
def evaluate(model, dataloader,criterion,device):
        
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels=[]
        all_predictions=[]
        with torch.no_grad():
            for text, labels in dataloader:
                text, labels = text.to(device), labels.to(device)
                outputs = model(text)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        f1_score_macro=f1_score(all_labels,all_predictions,average='weighted')
        return accuracy, avg_loss ,f1_score_macro

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RNN',choices=['RNN','GRU','BI-GRU','CNN'], help='Model to use for training')
    args = parser.parse_args()
    
    
    #Hyperparameters
    max_seq_length = 25 # use 50 for electronics dataset and 10 for SST2 dataset
    input_dim = 300
    hidden_dim = 100
    output_dim = 2 # 2 for SST2 dataset and 5 for electronics dataset   
    num_epochs = 20
    learning_rate = 0.002 # use 0.001 for electronics dataset and 0.0025 for SST2 dataset
    batch_size = 256 # use 32 for electronics dataset and 1024 for SST2 dataset
    
    #Load the preprocessed data
    # X,y = load_data('data.csv', max_seq_length)
    
    X,y = load_data('data/SST2/train.csv', max_seq_length,label_shifting=False)
    

    # Create the train dataloader
    train_dataloader = get_dataloader(X, y, batch_size=batch_size)
    
    X,y = load_data('data/SST2/validation.csv', max_seq_length,label_shifting=False)
    val_dataloader = get_dataloader(X, y, batch_size=batch_size)
    
    
    print("Creating the model")
    
    # Create the model
    print(f'Model selected: {args.model}')
    
    if args.model == 'RNN':
        
        model = RNNClassifier(input_dim, hidden_dim, output_dim).to(device)
        
    elif args.model == 'BI-GRU':
        model = Bi_GRUClassifier(input_dim, hidden_dim, output_dim).to(device)
        
    elif args.model == 'CNN':
        
        model = CNNClassifier(input_dim, output_dim, num_filters=100, filter_sizes=[3,4,5]).to(device)
        
    else:
        model = GRUClassifier(input_dim, hidden_dim, output_dim).to(device)
        
    
   
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print("Training the model")
    train(model,train_dataloader,val_dataloader,criterion,optimizer,device,num_epochs=num_epochs)
    
    
    # Save the model
    
    