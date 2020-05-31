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

from FutureNetArchitecture import FutureNet
from dataset import FutureDataSet
from utils import saveLog

import argparse
import pdb
import math

class Runner():
    def __init__(self, trainingData, gpuOn=False, epochs=200, lr=0.001, weight_decay=0.01):

        #instantiate the LSTM and hyperparams
        self.net = FutureNet()
        self.gpuOn = gpuOn
        self.lr = lr
        self.weight_decay = weight_decay

        #Move the network to the GPU if enabled
        if self.gpuOn:
            self.net = self.net.cuda()

        #Load Data & put into training DataLoader, which is used in train()
        train = FutureDataSet(trainingData)
        self.trainLoader = DataLoader(train, batch_size=1,
                            shuffle=True, num_workers=4)

    def train(self):

        #Print out the current network architecture with some border bars for classiness
        print("-------------------------------------")
        print(self.net)
        print("-------------------------------------")

        #Instantiate the loss function, optimizer, learning rate scheduler, and start recording training time
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = .25, patience = 5, threshold = .002, verbose = True, min_lr = [.000001])
        start = time.time()

        #The epoch training loop
        prev_loss = 0.0
        flat_count = 0
        self.net.train()
        for epoch in (self.epochs):
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
                    if self.gpuOn:
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
        torch.save(self.net, "Future_Model.pth")

if __name__ == '__main__':

    #Command Line Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--trainData')
    parser.add_argument('-rn', '--runName')
    args = parser.parse_args()

    #Initialize Runner obj and run training cycle
    #needs: trainingData - dataframe of all training data
    futureNet = Runner(trainingData)
    futureNet.train()

#Save log, diabled temporarily until review is finished
#saveLog(log_Path, iden, experiment['datapath'], net.arch_Name, finalEpoch, true, pred, net.seed, elapsed, str(args.runName), net)

