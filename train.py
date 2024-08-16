# # Let us write the training loop here

# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from dataloader import get_dataloader
# from model import RNNClassifier
# from eval import evaluate

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def train(model,dataloader,criterion,optimizer,device,num_epochs=10):
    
#     model.train()
#     model.to(device)
#     for epoch in range(num_epochs):
#         model.train()
#         for text, labels in dataloader:
#             text, labels = text.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(text)
            
#             loss = criterion(outputs, labels)
            
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}/{num_epochs} Loss: {loss.item()}')
        
#         accuracy = evaluate(model, dataloader,device)
#         print(f'Accuracy: {accuracy}')
#         # print(f'Report: {report}')
        
#     return model

# if __name__ == '__main__':
#     dataloader = get_dataloader('data_preprocessed.csv')
#     model = RNNClassifier(300, 128, 5).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=0.0025)
#     model = train(model,dataloader,criterion,optimizer,device,num_epochs=100)
#     torch.save(model.state_dict(), 'model.pth')
    

import torch
import torch.nn as nn
from torch.optim import Adam
from dataloader import get_dataloader
from model import RNNClassifier
from eval import evaluate
from preprocessing import load_data

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
        
    torch.save(model.state_dict(), 'model.pth')
    
    print("Model trained and saved successfully")
    
    

if __name__ == '__main__':
    
    #Hyperparameters
    max_seq_length = 50
    input_dim = 300
    hidden_dim = 128
    output_dim = 5
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 128
    
    #Load the preprocessed data
    X,y = load_data('data.csv', max_seq_length)
    
    print(X[0],y[0])
    

    
    # Create the dataloader
    dataloader = get_dataloader(X, y, batch_size=batch_size)
    
    
    print("Creating the model")
    
    # Create the model
    model = RNNClassifier(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print("Training the model")
    train(model,dataloader,criterion,optimizer,device,num_epochs=num_epochs)
    
    
    # Save the model
    
    