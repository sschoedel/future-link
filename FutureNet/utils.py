import numpy as np
from mini_lambda import x, is_mini_lambda_expr
from datetime import date
from datetime import datetime
import time

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix as conf

def saveLog(log_Path, iden, file_Path, arch_Name, epochs, true, pred, seed, elapsed, short, net):
    #Collecting all the relevant data for the log
    today = date.today().strftime("%b-%d-%y")
    timeT = datetime.now().strftime("%H:%M:%S")
    timeTitle = datetime.now().strftime("%H.%M.%S")

    testScore = accuracy_score(true, pred)
    F1Score = f1_score(true, pred, average = None)
    cm = conf(true, pred)

    if log_Path is None:
        log_Path = ""



    #Setting the path for the file to be written to
    fullPath = log_Path + iden + "_" + short + "_" + "CNN_Training_Log" + today + "_" + timeTitle + ".txt"

    #Write the log 
    with open(fullPath , 'w+') as file:
        file.write(today)
        file.write("\t" + timeT)
        file.write("\n")
        file.write("\n")
        file.write("File Used: ")
        file.write(str(file_Path))
        file.write("\n")
        file.write("Time Elapsed: ")
        file.write(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        file.write("\n")
        file.write("Epochs: " + str(epochs))
        file.write("\n")
        file.write("Architecture Used: " + arch_Name)
        file.write("\n")
        file.write("\n")
        file.write(str(net))
        file.write("\n")
        file.write("\n")
        file.write("Seed: " + str(seed))
        file.write("\n")
        file.write("Test Accuracy: " + str(testScore))
        file.write("\n")
        file.write("F1 Scores: " + str(F1Score))
        file.write("\n")
        file.write("Confusion Matrix: \n")
        np.savetxt(file, cm, fmt='%s\t')

def loadData(filePath):

        #Open File
        with open(filePath, 'r') as file:
            #Process first line of the file
            lines = file.readlines()
            firstLine = lines[0].split(",")
            X = np.array(firstLine[0:(len(firstLine)-1)], dtype=np.float32)
            Y = np.array(firstLine[(len(firstLine)-1):(len(firstLine))], dtype = np.int16)
            
            #Remove first line & set line count to 1
            lines.pop(0)
            count = 1
            
            #Process each line and tack it on to the bottom of the X and Y datasets
            for line in lines:
                count += 1
                nextLine = line.split(",")

                if len(nextLine) != len(firstLine):
                    raise ValueError("Strange data formatting at line " + str(count))
                
                subX = np.array(nextLine[0:(len(nextLine)-1)], dtype=np.float64)
                subY = np.array(nextLine[(len(nextLine)-1):(len(nextLine))], dtype=np.int16)

                X = np.vstack((X, subX))
                Y = np.hstack((Y, subY))

        return X, Y 