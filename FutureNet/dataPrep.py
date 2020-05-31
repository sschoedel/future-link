import pandas as pd
import numpy as np

import csv
from itertools import chain
import pdb

def get_mulList(*args):
    return map(list,zip(*args))

csv_data = open('NO2_data.csv','r')
data = list(csv.reader(csv_data))
keys = []
values = []
for key in data[0]:
    keys.append(key.lower())
    
for values in get_mulList(*data[1:]):
    valueList = []
    for value in values:
        valueList.append(float(value))
    values.append(valueList)

ind_dict = dict(zip(keys,values))

print(ind_dict)
pdb.set_trace()