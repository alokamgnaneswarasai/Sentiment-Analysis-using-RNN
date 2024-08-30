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
    
    train_losses=[]
    train_accuracies=[]
    val_losses=[]
    val_accuracies=[]
    max_accuracy=0
    
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
        
        print('**'*25,'Epoch',epoch+1,'**'*25)
        # print(f'Epoch {epoch+1}/{num_epochs} Loss: {loss.item()}')
        train_accuracy, train_loss ,_= evaluate(model, train_dataloader,criterion,device)
        val_accuracy, val_loss ,fl_score= evaluate(model, val_dataloader,criterion,device)
        max_accuracy=max(max_accuracy,val_accuracy)
        print(f'Train Accuracy: {train_accuracy} Train Loss: {train_loss}')
        print(f'validation Accuracy: {val_accuracy} validation Loss: {val_loss}')
        print(f'F1 Score: {fl_score}')
        print(f'Max Accuracy acheived till now: {max_accuracy}')
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        
        
        # Add L2 regularization
        # for param in model.parameters():
        #     param.data = param.data.renorm(p=2, dim=0, maxnorm=3)
        
    plot_loss_accuracy('plots/',train_losses,train_accuracies,val_losses,val_accuracies)
    
 
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
    
    
def plot_loss_accuracy(path,train_loss,train_accuracy,val_loss,val_accuracy):
    
    import matplotlib.pyplot as plt
    plt.plot(train_loss,label='Train Loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(path+'loss.png')
    plt.show()
    
    plt.plot(train_accuracy,label='Train Accuracy')
    plt.plot(val_accuracy,label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(path+'accuracy.png')
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RNN',choices=['RNN','GRU','BI-GRU','CNN'], help='Model to use for training')
    args = parser.parse_args()
    
    
    #Hyperparameters
    train_max_seq_length = 30 # use 50 for electronics dataset and 10 for SST2 dataset
    valid_max_seq_length = 40
    input_dim = 300
    hidden_dim = 100
    output_dim = 2 # 2 for SST2 dataset and 5 for electronics dataset   
    num_epochs = 256
    learning_rate = 0.0015 # use 0.001 for electronics dataset and 0.0025 for SST2 dataset
    batch_size = 64# use 32 for electronics dataset and 1024 for SST2 dataset
    
    #Load the preprocessed data
    # X,y = load_data('data.csv', train_max_seq_length)
    
    # X,y = load_data('data/electronics/train.csv', train_max_seq_length)
    
    X,y = load_data('data/SST2/train.csv', train_max_seq_length,label_shifting=False)
    
    # Create the train dataloader
    train_dataloader = get_dataloader(X, y, batch_size=batch_size)
    
    # X,y = load_data('data/SST2/validation.csv', valid_max_seq_length,label_shifting=False)
    # X,y = load_data('electronics_validation.csv', valid_max_seq_length)
    X,y = load_data('data/SST2/testdata.csv', valid_max_seq_length,label_shifting=False)
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
    # use adadealta optimizer as there is no need to set initial learning rate 
    # optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95, eps=1e-06, weight_decay=0)
    
    
    print("Training the model")
    train(model,train_dataloader,val_dataloader,criterion,optimizer,device,num_epochs=num_epochs)
    

    
    