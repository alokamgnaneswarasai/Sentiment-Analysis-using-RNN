import torch
import torch.nn as nn
from torch.optim import Adam
from dataloader import get_dataloader
from model import RNNClassifier, GRUClassifier , Bi_GRUClassifier , CNNClassifier
from preprocessing import load_data
import argparse
from sklearn.metrics import f1_score

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')



def train(model,train_dataloader,val_dataloader,criterion,optimizer,device,num_epochs,args):
    
    train_losses=[]
    train_accuracies=[]
    val_losses=[]
    val_accuracies=[]
    max_accuracy=0
    max_f1_score=0
    
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
        if val_accuracy>max_accuracy:
            # save with file name wich consists of all args
            torch.save(model.state_dict(),f'models/{args.dataset}/{args.model}_model_{args.batch_size}_{args.num_epochs}_{args.learning_rate}_{args.input_dim}_{args.hidden_dim}_{val_accuracy}.pth')
            
         
            import os
            if max_accuracy!=0:
                os.remove(f'models/{args.dataset}/{args.model}_model_{args.batch_size}_{args.num_epochs}_{args.learning_rate}_{args.input_dim}_{args.hidden_dim}_{max_accuracy}.pth')
            
            
        max_accuracy=max(max_accuracy,val_accuracy)
        max_f1_score=max(max_f1_score,fl_score)
        print(f'Train Accuracy: {train_accuracy} Train Loss: {train_loss}')
        print(f'validation Accuracy: {val_accuracy} validation Loss: {val_loss}')
        print(f'F1 Score: {fl_score}')
        print(f'Max Accuracy acheived till now: {max_accuracy}')
        print(f'Max F1 Score acheived till now: {max_f1_score}')
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        
        
        # Add L2 regularization
        # for param in model.parameters():
        #     param.data = param.data.renorm(p=2, dim=0, maxnorm=3)
        
    plot_loss_accuracy('plots/',train_losses,train_accuracies,val_losses,val_accuracies)
    
 
    # torch.save(model.state_dict(), f'models/{args.model}_model.pth')
    # torch.save(model.state_dict(),f'models/{args.dataset}/{args.model}_model.pth')
    
    print("Model trained and saved successfully in path ",f"models/{args.dataset}/{args.model}_model_{args.batch_size}_{args.num_epochs}_{args.learning_rate}_{args.input_dim}_{args.hidden_dim}_{max_accuracy}.pth")
    
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
        # f1_score_macro=f1_score(all_labels,all_predictions,average='weighted')
        # calculate f1_score_mini
        f1_score_mini=f1_score(all_labels,all_predictions,average='micro')
        return accuracy, avg_loss ,f1_score_mini
    

    
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
    parser.add_argument('--model', type=str, default='CNN',choices=['RNN','GRU','BI-GRU','CNN'], help='Model to use for training')
    parser.add_argument('--dataset', type=str, default='electronics',choices=['SST2','electronics'], help='Dataset to use for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--input_dim', type=int, default=300, help='Input dimension for the model')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension for the model')
    parser.add_argument('--test', type=bool, default=False, help='Whether to train the model or test the model')
    args = parser.parse_args()
    
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    if args.dataset == 'SST2':
        train_max_seq_length = 30
        valid_max_seq_length = 40
        output_dim = 2
        X,y = load_data('data/SST2/train.csv', train_max_seq_length,label_shifting=False)
        train_dataloader = get_dataloader(X, y, batch_size=args.batch_size)
        # X,y = load_data('data/SST2/testdata.csv', valid_max_seq_length,label_shifting=False)
        X,y = load_data('data/SST2/validation.csv', valid_max_seq_length,label_shifting=False)
        val_dataloader = get_dataloader(X, y, batch_size=args.batch_size)
        
      
    else:
        train_max_seq_length = 60
        valid_max_seq_length = 60
        output_dim = 5 
        X,y = load_data('data.csv', train_max_seq_length,label_shifting=True)
        # X,y = load_data('data_augmented.csv', train_max_seq_length,label_shifting=True)
        # X,y = load_data('data/electronics/train.csv', train_max_seq_length,label_shifting=True)
        train_dataloader = get_dataloader(X, y, batch_size=args.batch_size)
        # X,y = load_data('electronics_validation.csv', valid_max_seq_length,label_shifting=True)
        X,y = load_data('amazon_reviews.csv', valid_max_seq_length,label_shifting=True)
        val_dataloader = get_dataloader(X, y, batch_size=args.batch_size)
    
    
    #Hyperparameters
    # train_max_seq_length = 30 # use 50 for electronics dataset and 10 for SST2 dataset
    # valid_max_seq_length = 40
    # input_dim = 300
    # hidden_dim = 100
    # output_dim = 2 # 2 for SST2 dataset and 5 for electronics dataset   
    # num_epochs = 256
    # learning_rate = 0.0015 # use 0.001 for electronics dataset and 0.0025 for SST2 dataset
    # batch_size = 64# use 32 for electronics dataset and 1024 for SST2 dataset
    
    #Load the preprocessed data
    # X,y = load_data('data.csv', train_max_seq_length)
    
    # X,y = load_data('data/electronics/train.csv', train_max_seq_length)
    
    # X,y = load_data('data/SST2/train.csv', train_max_seq_length,label_shifting=False)
    
    # # Create the train dataloader
    # train_dataloader = get_dataloader(X, y, batch_size=batch_size)
    
    # # X,y = load_data('data/SST2/validation.csv', valid_max_seq_length,label_shifting=False)
    # # X,y = load_data('electronics_validation.csv', valid_max_seq_length)
    # X,y = load_data('data/SST2/testdata.csv', valid_max_seq_length,label_shifting=False)
    # val_dataloader = get_dataloader(X, y, batch_size=batch_size)
    
    
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
    
    train(model,train_dataloader,val_dataloader,criterion,optimizer,device,num_epochs=num_epochs,args=args)
    

    
    