from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import datereaderYE

class FutureDataSet(Dataset):

    def __init__(self, dataframe, y):
        self.dataframe = dataframe
        self.y = y

    #Return how many samples are in the dataset
    def __len__(self):
        return len(self.dataframe)

    #Retrieve and prepare the data to be used for one sample
    def __getitem__(self, idx):
        
        stateLine = self.dataframe.iloc[idx, 0:]
        state = np.array([stateLine[0:]])[0]
        #print(state.shape)
        state = np.array([stateLine[1:]])[0]
        #print(state.shape)
        gasVals = self.y
        #print(gasVals.shape)
        state = np.hstack((state, gasVals[:len(gasVals)-1]))
        
        state = state.astype('float')
        label = gasVals[1:]
        return state, label

if __name__ == '__main__':
    datas = []
    for state in ['virginia', 'wyoming', 'kansas', 'colorado']:
        #print(state)
        labels, output = datereaderYE.get_X("2020-03-22", "2020-05-28", state)
        datas.append(output)
    testData = pd.DataFrame.from_records(datas)


    y = np.random.rand(66 + 1).reshape(((66 + 1),1))
    data = FutureDataSet(testData, y)
    print(data.__getitem__(2))
