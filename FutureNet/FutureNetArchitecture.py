import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FutureNet(nn.Module):
    
    def __init__(self, input_dim=7, hidden_dim=50, batch_size, output_dim=1, num_layers=2):
        super(FutureNet, self).__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        #define LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        #define output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        #initialize the hidden state
        #each LSTM cell needs a hidden state from a previous cell, but the first cell
        #needs this initialization instead
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        
    def forward(self, input):
        #forward pass
        #shape of lstm_out: [input_size, batch_size, hidden_dim]
        #shape of self.hidden: (a, b) where a and b have shape (num_layers, batch_size, hidden_dim)

        #reshape the input to 3D and feed to lstm layer
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        #only can take the output from the final timestep
        #can pass on the entirety of lstm_out to the next layer if it is a seq2seq predection
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

