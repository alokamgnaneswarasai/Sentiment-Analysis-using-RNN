# # Lets create the dataloader for the text classification task here

# import torch
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np

# class TextDataset(Dataset):
#     def __init__(self, data_path):
#         self.data = pd.read_csv(data_path)
#         self.labels = torch.tensor(self.data['overall'].values-1, dtype=torch.long)
#         # self.text = torch.tensor(self.data['reviewText'].apply(lambda x: np.fromstring(x[1:-1], sep=' ', dtype=np.float32)).tolist(), dtype=torch.float32)
#         # self.text = torch.tensor(self.data['reviewText'].apply(lambda x: np.array(x[1:-1].split(), dtype=np.float32)).tolist(), dtype=torch.float32)
#         self.text = torch.tensor(np.array(self.data['reviewText'].apply(lambda x: np.array(x[1:-1].split(), dtype=np.float32)).tolist()), dtype=torch.float32)
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.text[idx], self.labels[idx]
    
# def get_dataloader(data_path, batch_size=32, shuffle=True):
#     dataset = TextDataset(data_path)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader

# if __name__ == '__main__':
#     dataloader = get_dataloader('data_preprocessed.csv')
#     for text, labels in dataloader:
#         print(text, labels)
#         break
        
    
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    
def get_dataloader(data, labels, batch_size=32, shuffle=True):
    dataset = TextDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
