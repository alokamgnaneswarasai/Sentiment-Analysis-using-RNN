
    
import torch
from sklearn.metrics import accuracy_score, classification_report,f1_score

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
        return accuracy, avg_loss, f1_score_macro
    