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
            out, hn = self.rnn(x, h0)
            
            # out = self.fc(out[:, -1, :]) # get the output of the last time step
            out = self.fc(hn[-1]) # get the output of the last time step by using the hidden state of the last time step
            return out
        
class GRUClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)
        
        out = self.fc(hn[-1])
        return out
    
    
        
    