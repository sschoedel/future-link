<<<<<<< HEAD:reader.py
import csv
import pdb
from datetime import datetime

def readIHMEFile(filename):
    #path = 'Summary_stats_all_locs.csv'

    masterDict = dict()

    with open(filename, 'r') as csv_file:
        for i, col in enumerate(csv_file):
            locTemp = col.lower().split(",")
            loc = []
            for item in locTemp:
                cleanItem = item.strip('\"').strip('\n')
                loc.append(cleanItem.strip('\"'))

            if i == 0:
                labels = loc
                labels.pop(0)
                
            else:
                #pdb.set_trace()
                valueDict = dict()
                name = loc[0]
                loc.pop(0)
                for i, item in enumerate(loc):
                    
                    if item[0:4] == '2020':
                        item = datetime.strptime(item.strip('\"').strip('\n'), '%Y-%m-%d')

                    try:
                        item = int(item)
                    except:
                        pass

                    if item == '':
                        item = None

                    


                    valueDict.update({labels[i]: item})
                masterDict.update({name: valueDict})
    #print(masterDict)
    return(masterDict)

if __name__ == '__main__':
    masterDict = readIHMEFile('Summary_stats_all_locs.csv')
=======
import csv
import pdb
from datetime import datetime

def readIHMEFile(filename):
    #path = 'Summary_stats_all_locs.csv'

    masterDict = dict()

    with open(filename, 'r') as csv_file:
        for i, col in enumerate(csv_file):
            locTemp = col.lower().split(",")
            loc = []
            for item in locTemp:
                loc.append(item.strip('\"').strip('\n'))

            if i == 0:
                labels = loc
                labels.pop(0)
                
            else:
                #pdb.set_trace()
                valueDict = dict()
                name = loc[0]
                loc.pop(0)
                for i, item in enumerate(loc):
                    
                    if item[0:4] == '2020':
                        item = datetime.strptime(item.strip('\"').strip('\n'), '%Y-%m-%d')

                    try:
                        item = int(item)
                    except:
                        pass

                    if item == '':
                        item = None

                    


                    valueDict.update({labels[i]: item})
                masterDict.update({name: valueDict})
    #print(masterDict)
    return(masterDict)

if __name__ == '__main__':
    masterDict = readIHMEFile('Summary_stats_all_locs.csv')
>>>>>>> 34e0f7b568fe8ab84bd520fca9905121f5d4cc9b:python/reader.py
    print(masterDict['wyoming']['peak_bed_day_mean'].date())