import csv
import pdb
from datetime import datetime, timedelta
import numpy as np
import dataLoad
import pandas as pd
import pdb

from itertools import chain
import joblib as jb

def get_X(start_date, end_date, state_name):
    start_date = datetime.strptime(start_date.strip('\"').strip('\n'), '%Y-%m-%d')
    end_date = datetime.strptime(end_date.strip('\"').strip('\n'), '%Y-%m-%d')
    input_dict = dataLoad.getIHMEData()
    features = ['travel_limit_start_date', 'travel_limit_end_date', 'stay_home_start_date', 'stay_home_end_date', 'educational_fac_start_date', 
    'educational_fac_end_date', 'any_gathering_restrict_start_date', 'any_gathering_restrict_end_date', 'any_business_start_date',
    'any_business_end_date', 'all_non-ess_business_start_date', 'all_non-ess_business_end_date']
    #print(input_dict)
    location_data = input_dict.get(state_name)
    labels = list(location_data)
    xPol = []
    curr_day = start_date
    for day in range((end_date - start_date).days):
        output_Arr = []
        for i, feat in enumerate(features):
            if "start" in feat:
                item = location_data[feat]
                if item == None:
                    output_Arr.append(0)
                elif(curr_day < item):
                    output_Arr.append(0)
                elif(location_data[features[i+1]] == None):
                    output_Arr.append(1)
                elif(curr_day > location_data[features[i+1]]):
                    output_Arr.append(0)
                else:
                    output_Arr.append(1)

        xPol.append(output_Arr)
        curr_day += timedelta(days = 1)
    colLab = [feat[:len(feat) - 11] for i, feat in enumerate(features) if i % 2 == 0]
        
    return pd.DataFrame(data = np.array(xPol), columns = colLab)

def get_mulList(*args):
    return map(list,zip(*args))

def get_Ydict():   
    csv_data = open('NO2_data.csv','r')
    data = list(csv.reader(csv_data))
    ind_dict = dict(zip([key.lower() for key in data[0]],[[abs(float(value)) for value in values] for values in get_mulList(*data[1:])]))

    return ind_dict, list(ind_dict)

def get_XY(end_d, start_d = '2020-02-22'):
    y_dict, states = get_Ydict()
    start_date = datetime.strptime(start_d, '%Y-%m-%d')
    end_date = datetime.strptime(end_d, '%Y-%m-%d')
    num_days = (end_date - start_date).days
    X = []
    Y = []
    stateOrder = []
    for state in states:
        print(state)
        y = y_dict[state][:num_days]
        x = get_X(start_d, end_d, state)
        #pdb.set_trace()
        X.append(x)
        Y.append(y)
        stateOrder.append(state)

    return X, Y, stateOrder


if __name__ == '__main__':
    x, y, states = get_XY('2020-05-28')
    print(len(x))
    dumpObj = [x,y,states]
    print(dumpObj)
    jb.dump(dumpObj, "lstm_Data.joblib")
    #jb.dump(y, "y_NO2.joblib")
