import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from yaml import load, dump
try: 
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError: 
    from yaml import Loader, Dumper

import joblib as jb
import argparse

from utils import loadData

#Function to load all of the data from different datasets and combine it together horizontally
def loadAllData(datasets):
    print(datasets)
    x, y = loadData(datasets[0])
    datasets.pop()
    if len(datasets) != 0:
        for dataset in datasets:
            xTemp, yTemp = loadData(dataset)
            for i, yy in enumerate(y):
                if y[i] != yy:
                    raise ValueError("labels are not correspondingly ordered at line: " + str(i) )
            x = np.append(x,xTemp,axis=1)
            print(x.shape) 

    return x, y

#Command Line Parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-o', '--output')
parser.add_argument('-sn', '--setName')
args = parser.parse_args()

if args.config is None:
    config_path = 'experiment_config.yml'
else:
    config_path = args.config

#Load the experiment and architecture config files
experiment = load(open(config_path), Loader=Loader)
arch = load(open(experiment['architecture_path']), Loader=Loader)

#Load all of the datasets described in the experiment config
datasets = experiment['datapath']
x, y = loadAllData(datasets)

#Preprocess the data with TruncatedSVD into a number of components set by the architecture config file.
seed = 42
svd = TruncatedSVD(n_components = arch['n_components'], n_iter = 7 , random_state = seed)
x = svd.fit_transform(x)

#Splits the data into stratified, randomized training and testing sets
trainX, testX, trainY, testY = train_test_split(
x, y, train_size = .7, random_state = seed, stratify = y)

#Put the loaded data together into a Panda Dataframe
trainingData = pd.DataFrame.from_records(np.append(trainX, np.reshape(trainY,(len(trainY),1)), axis= 1))
testingData = pd.DataFrame.from_records(np.append(testX, np.reshape(testY,(len(testY),1)), axis=1))

#Save the prepared dataframes
jb.dump(trainingData, args.setName + "_trainingData.joblib")
jb.dump(testingData, args.setName + "_testingData.joblib")
