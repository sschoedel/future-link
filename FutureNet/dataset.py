from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import datereaderYE

import joblib as jb

class FutureDataSet(Dataset):

    def __init__(self, dataframe, y):
        self.dataframe = dataframe
        self.y = y

    #Return how many samples are in the dataset
    def __len__(self):
        return len(self.dataframe)

    #Retrieve and prepare the data to be used for one sample
    def __getitem__(self, idx):
        
        stateLine = self.dataframe[idx]
        #print(state.shape)
        state = np.array(stateLine[1:])
        #print(state.shape)
        gasVals = np.array(self.y[idx]) / float(1E+20)
        #print(gasVals.shape)
        #print(state)
        #print(gasVals.shape)
        state = np.hstack((state, gasVals.reshape((len(gasVals),1))[:len(gasVals)-1]))
        
        state = state.astype('float')
        label = gasVals[1:]
        return state, label

if __name__ == '__main__':

    allData = jb.load('lstm_Data.joblib')
    x = allData[0]
    y = allData[1]

    data = FutureDataSet(x, y)
    print("--------------------------")
    print(data.__getitem__(2))
