# import torch
# from sklearn.metrics import accuracy_score, classification_report

# def evaluate(model, dataloader,device):
    
#     model.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for text, labels in dataloader:
#             text, labels = text.to(device), labels.to(device)
#             outputs = model(text)
#             _, predicted = torch.max(outputs, 1)
#             y_true += labels.tolist()
#             y_pred += predicted.tolist()
            
#     accuracy = accuracy_score(y_true, y_pred)
#     # report = classification_report(y_true, y_pred)
    
#     return accuracy

# if __name__ == '__main__':
    
#     from dataloader import get_dataloader
#     from model import RNNClassifier
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataloader = get_dataloader('data_preprocessed.csv')
#     model = RNNClassifier(300, 128, 5).to(device)
#     model.load_state_dict(torch.load('model.pth'))
#     accuracy, report = evaluate(model, dataloader,device)
    
#     print(f'Accuracy: {accuracy}')
#     print(f'Report: {report}')
    
    
import torch
from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, dataloader,criterion,device):
        
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for text, labels in dataloader:
                text, labels = text.to(device), labels.to(device)
                outputs = model(text)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        return accuracy, total_loss / len(dataloader)
    