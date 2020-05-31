import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from yaml import load, dump
try: 
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError: 
    from yaml import Loader, Dumper

class FutureNet(nn.Module):

    def __init__(self, architecture_filepath):
        super(FutureNet, self).__init__()

        self.seed = torch.initial_seed()
        print(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ###Read the yaml dictionary containing the architecture
        with open(architecture_filepath) as file:
            self.architecture = load(file, Loader=Loader)

            self.hidden_dim = self.architecture['input_width']

            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.lstm = nn.LSTM(self.architecture['input_width'], self.hidden_dim)

            # The linear layer that maps from hidden state space to tag space
            num_classes = 7
            self.hidden2tag = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, sentence):
        print(sentence)
        print(sentence.size())
        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        print("------------------------------------")
        print(lstm_out)
        print(lstm_out.size())
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        print("------------------------------------")
        print(tag_space)
        print(tag_space.size())
        tag_scores = F.log_softmax(tag_space, dim=7)
        return tag_scores