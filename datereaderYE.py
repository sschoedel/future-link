import csv
import pdb
from datetime import datetime, timedelta
import numpy as np
import dataLoad
import pandas as pd

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
        
    return features, pd.DataFrame(data = np.array(xPol), columns = colLab)

if __name__ == '__main__':
    labels, output = get_X("2020-02-22", "2020-05-28", 'virginia')
    print(labels)
    print(output)






