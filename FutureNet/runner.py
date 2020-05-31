import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix as conf, f1_score

import time
from datetime import date
from datetime import datetime

from yaml import load, dump
try: 
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError: 
    from yaml import Loader, Dumper

from FutureNetArchitecture import FutureNet
from dataset import TrussDataSet
from utils import saveLog

import argparse
import pdb
import math
import joblib as jb

class Runner():
    def __init__(self, architecture, trainingData):

        #instantiate the architecture, experiment config, and neural network class variables
        self.arch = architecture
        self.net = FutureNet()
        #Move the network to the GPU if enabled
        if experiment['gpuOn']:
            self.net = self.net.cuda()

        #Load Data & put into training DataLoader, which is used in train()
        train = TrussDataSet(trainingData, xWidth = arch['input_width'], yWidth = arch['input_height'])
        self.trainLoader = DataLoader(train, batch_size= experiment['batch_size'],
                            shuffle=True, num_workers=4)

    def train(self):

        #Print out the current network architecture with some border bars for classiness
        print("-------------------------------------")
        print(self.net)
        print("-------------------------------------")

        #Instantiate the loss function, optimizer, learning rate scheduler, and start recording training time
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(self.net.parameters(), lr = self.arch['lr'], weight_decay= self.arch['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = .25, patience = 5, threshold = .002, verbose = True, min_lr = [.000001])
        start = time.time()

        #The epoch training loop
        prev_loss = 0.0
        flat_count = 0
        self.net.train()
        for epoch in range(self.experiment['epochs']):
            print("Epoch #:" + str(epoch))
            running_loss = 0.0

            for i, data in enumerate(self.trainLoader, 0):
                #inputs: 2d array, full time sequence
                #labels: 1d array
                inputs, labels = data
            
                #Zero the gradient of the NN before the training loop begins
                optimizer.zero_grad()

                #need to feed in data one row at a time
                for day in inputs:
                    #Move the loaded data to the GPU if enabled
                    if experiment['gpuOn']:
                        inputs = inputs.float().cuda()
                        labels = labels.float().cuda()
                    else:
                        inputs = inputs.float()
                        labels = labels.float()
                
                    #Predict from the current model, calculate the loss, and perform backprop
                    output = self.net(day)
                    loss = criterion(output,labels)
                    loss.backward()
                    optimizer.step()

                    #Keep a running loss total throughout the epoch for reporting reasons
                    running_loss += loss.item()

            #Update the learning rate scheduler
            scheduler.step(running_loss)

            #Early stop mechanic: If the loss is greater than the previous loss or fluctuates within the threshold in reference to the previous stop, increase the flat count
            stop_thres = .0005
            if np.round(np.abs(running_loss/(i+1) - prev_loss/(i+1)), decimals = 3) <= stop_thres or np.round(running_loss/(i+1), decimals = 3) >= np.round(prev_loss/(i+1), decimals = 3):
                flat_count = flat_count + 1
            else:
                flat_count = 0

            print('[{}] Loss: {:.4f} | Flatline Count: {}'.format(epoch, running_loss/(i+1), flat_count))

            #If flat count is greater than stop_count and the learning rate scheduler has reached its minimum lr, cut off the training of the network and record the final epoch
            finalEpoch = epoch
            stop_count = 10
            if flat_count >= stop_count and optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
                break

            #Record the running_loss for the early stop mechanic, then zero out the running loss
            prev_loss = running_loss
            running_loss = 0.0

        #Print the elapsed time and celebrate that a trained network has been made!
        elapsed = time.time() - start      
        print("Elapsed Time: {:.3f}".format(elapsed))
        print('Finished Training')

        #Save the trained network
        torch.save(self.net, args.runName + "_Model.pth")

if __name__ == '__main__':

    #Command Line Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-td', '--trainData')
    parser.add_argument('-rn', '--runName')
    args = parser.parse_args()

    if args.config is None:
        config_path = 'experiment_config.yml'
    else:
        config_path = args.config

    #load config, architecture, testing Data, and NN files
    experiment = load(open(config_path), Loader=Loader)
    arch = load(open(experiment['architecture_path']), Loader=Loader)
    #trainingData = jb.load(args.trainData)
    torch = 
    #Initialize Runner obj and run training cycle
    trussNet = Runner(experiment, arch, trainingData)
    trussNet.train()

#Save log, diabled temporarily until review is finished
#saveLog(log_Path, iden, experiment['datapath'], net.arch_Name, finalEpoch, true, pred, net.seed, elapsed, str(args.runName), net)

