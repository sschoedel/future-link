import pandas as pd
import numpy as np

import csv
from itertools import chain
import pdb

def get_mulList(*args):
    return map(list,zip(*args))

def get_Ydict():   
    csv_data = open('NO2_data.csv','r')
    data = list(csv.reader(csv_data))
    ind_dict = dict(zip([key.lower() for key in data[0]],[[float(value) for value in values] for values in get_mulList(*data[1:])]))

    print(ind_dict)
    pdb.set_trace()