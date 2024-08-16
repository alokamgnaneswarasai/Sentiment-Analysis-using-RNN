# # Lets define the RNN model for text classification task here

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class RNNClassifier(nn.Module):
    
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(RNNClassifier, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, x):
#         out, _ = self.rnn(x)
#         out = self.fc(out)
#         return out
    
# if __name__ == '__main__':
#     model = RNNClassifier(300, 128, 5)
#     print(model)
    

import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
        
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(RNNClassifier, self).__init__()
            self.hidden_dim = hidden_dim
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            
            h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.rnn(x, h0)
            out = self.fc(out[:, -1, :]) # get the output of the last time step
            return out
        
    