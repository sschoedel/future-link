from torch.utils.data import Dataset
import numpy as np
import pandas
import datereaderYE

class FutureDataSet(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transform

    #Return how many samples are in the dataset
    def __len__(self):
        return len(self.dataframe)

    #Retrieve and prepare the data to be used for one sample
    def __getitem__(self, idx):
        
        stateLine = self.dataframe.iloc[idx, 0:]
        state = np.array([stateLine[0:(len(stateLine)-1)]])
        label = stateLine[(len(stateLine)-1)]
        
        state = state.astype('float')
        return state, label

if __name__ == '__main__':
    datas = []
    for state in ['virginia, wyoming, kansas, colorado']:
        labels, output = datereaderYE.get_X("2020-03-22", "2020-05-28", state)
        datas.append(output)
    testData = pd.DataFrame.from_records(datas)

    data = FutureDataSet(testData)
    print(data.__getitem__(2))
