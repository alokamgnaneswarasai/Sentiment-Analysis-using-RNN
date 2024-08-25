import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class Bi_GRUClassifier(nn.Module):
        
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Bi_GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, x):
        
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)
        
        out = self.fc(torch.cat((hn[-2], hn[-1]), dim=1)) # concatenate the hidden states of the last time step from both directions
        return out
    
    
class CNNClassifier(nn.Module):
        
    def __init__(self,input_dim,output_dim,num_filters,filter_sizes,dropout=0.5):
        super(CNNClassifier,self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1,num_filters,(fs,input_dim)) for fs in filter_sizes
            ])
        self.fc = nn.Linear(num_filters*len(filter_sizes),output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        # x = [batch size, sent len, emb dim]
        x = x.unsqueeze(1) # x = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # conved_n = [batch size, num_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved] # pooled_n = [batch size, num_filters]
        cat = self.dropout(torch.cat(pooled,dim=1)) # cat = [batch size, num_filters * len(filter_sizes)]
        return self.fc(cat) # [batch size, output dim]
        
    