import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix as conf, f1_score

from yaml import load, dump
try: 
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError: 
    from yaml import Loader, Dumper

from future_net import FutureNet
from dataset import TrussDataSet

import argparse
import joblib as jb

class Tester():
    def __init__(self, architecture, experiment, testingData, net):

        #Save the architecture, experiment configs, and network as class variables
        self.arch = architecture
        self.experiment = experiment
        self.net = net

        #Load Data & put into testing DataLoader, which is used in evaluate()
        test = TrussDataSet(testingData, xWidth = self.arch['input_width'], yWidth = self.arch['input_height'])
        self.testLoader = DataLoader(test, batch_size=experiment['batch_size'],
                                shuffle=True, num_workers=4)

    def evaluate(self, ):

        #Run a testing loop until all of the testing data is used
        self.net.eval()
        with torch.no_grad():
            pred = []
            true = []
            for i, data in enumerate(self.testLoader, 0):
                inputs, labels = data

                #Put data onto the GPU if enabled
                if self.experiment['gpuOn']:
                    inputs = inputs.float().cuda()
                else:
                    inputs = inputs.float()

                #Run an inference on the network
                outputs, _ = self.net(inputs)

                #Move the prediction back onto the cpu & find the index of the greatest probability.
                outputs = outputs.cpu()
                outputs = np.argmax(outputs, 1)

                #Append the prediction and ground truths to a running list
                pred = np.hstack((pred, outputs.numpy()))
                true = np.hstack((true, labels.numpy()))

            #Run performance metrics & Print out the results
            testScore = accuracy_score(true, pred)
            F1Score = f1_score(true, pred, average = None)

            print(testScore)
            print(F1Score)
            cm = conf(true, pred)
            print(cm)

if __name__ == '__main__':

    #Command Line Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-td', '--testData')
    parser.add_argument('-rn', '--runName')
    parser.add_argument('-n', '--net')
    args = parser.parse_args()

    if args.config is None:
        config_path = 'experiment_config.yml'
    else:
        config_path = args.config

    #load config, architecture, testing Data, and NN files
    experiment = load(open(config_path), Loader=Loader)
    arch = load(open(experiment['architecture_path']), Loader=Loader)
    testingData = jb.load(args.testData)
    trussNet = torch.load(args.net)

    #Initialize Tester obj and run evaluation
    netTest = Tester(arch, experiment, testingData, trussNet)
    netTest.evaluate()