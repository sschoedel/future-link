from torch.utils.data import Dataset
import numpy as np

class TrussDataSet(Dataset):

    def __init__(self, dataframe, transform = None, xWidth = 6, yWidth = 6):
        self.dataframe = dataframe
        self.transform = transform
        self.xWidth = xWidth
        self.yWidth = yWidth

    #Return how many samples are in the dataset
    def __len__(self):
        return len(self.dataframe)

    #Retrieve and prepare the data to be used for one sample
    def __getitem__(self, idx):
        
        trussLine = self.dataframe.iloc[idx, 0:]
        truss = np.array([trussLine[0:(len(trussLine)-1)]])
        label = trussLine[(len(trussLine)-1)]
        
        truss = truss.astype('float').reshape(1, self.xWidth, self.yWidth)
        return truss, label