import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from yaml import load, dump
try: 
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError: 
    from yaml import Loader, Dumper

class TrussNet(nn.Module):

    def __init__(self, architecture_filepath):
        super(TrussNet, self).__init__()

        self.seed = torch.initial_seed()
        print(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ###Read the yaml dictionary containing the architecture
        with open(architecture_filepath) as file:
            self.architecture = load(file, Loader=Loader)

            if self.architecture['resnet']:
                layers = []
                inF = self.architecture['input_width']
                outF = self.architecture['num_classes']
                layers.append(nn.Linear(inF, inF))

                blockInF = inF
                for i in range(1, self.architecture['num_blocks'] + 1):
                    layers.append(self.add_resGrp(inF = blockInF,
                                                outF =  round(blockInF * self.architecture['resBlock_step']),
                                                dropP = self.architecture['res_drop']))
                    blockInF = round(blockInF * self.architecture['resBlock_step'])

                layers.append(nn.BatchNorm1d(blockInF))
                layers.append(nn.ReLU())
                layers.append(nn.Linear(blockInF, outF))
                self.res = nn.Sequential(*layers)

            else:
                self.conv_layers = []
                if self.architecture['lin_Only'] is False:
                    for _, params in self.architecture['conv_layers'].items():
                        self.conv_layers.append(nn.Conv2d(params['in_channels'], params['out_channels'], (params['conv_window_width'], params['conv_window_height']), 
                        dilation = params['dilation'], padding = params['padding'], stride=params['stride']))
                        self.conv_layers.append(nn.ReLU())

                self.lin_layers = []
                for _, params in self.architecture['lin_layers'].items():
                    if params['layerType'] == "dropout":
                        self.lin_layers.append(nn.Dropout(p = params['drop_Chance']))
                    else:
                        self.lin_layers.append(nn.Linear(params['in_channels'], params['out_channels']))
                        self.lin_layers.append(nn.ReLU())

                if self.architecture['lin_Only'] is False:
                    self.conv_layers = nn.Sequential(*self.conv_layers)
                self.lin_layers = nn.Sequential(*self.lin_layers)
                self.out = nn.Softmax(dim=1)

        self.arch_Name = self.architecture['architecture']

    def forward(self, x):
        
        if self.architecture['resnet']:
            x = x.view(x.size(0), -1)
            x = self.res(x)
            y = x

        else:
            if self.architecture['lin_Only'] is False:
                x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.lin_layers(x)
            y = self.out(x)
        
        return x,y

    def add_resGrp(self, inF, outF, dropP):
        blocks = []
        blocks.append(ResBlock(inF, outF, dropP))

        for i in range(1,self.architecture['blocks_per_group']):
            blocks.append(ResBlock(inF, outF, dropP))

        return nn.Sequential(*blocks)
        

class ResBlock(nn.Module):

    def __init__(self, inF, outF, dropP):
        super(ResBlock, self).__init__()
        self.inF = inF
        self.outF = outF

        self.shortcut = nn.Linear(inF, outF)
        self.start_norm = nn.Sequential(nn.BatchNorm1d(inF), nn.ReLU())
        self.layers = self.build_block(inF, outF, dropP)

    def build_block(self, inF, outF, dropP):
        layers = []
        layers.append(nn.Linear(inF, outF))
        layers.append(nn.BatchNorm1d(outF))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropP))
        layers.append(nn.Linear(outF, outF))

        return nn.Sequential(*layers)

    def forward(self, x):
        res = x

        x = self.start_norm(x)
        res = self.shortcut(x)
        x = self.layers(x)
        
        x = res + x

        return x