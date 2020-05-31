#python3 dataPrep.py -sn bash
python3 runner.py -rn bash_ResNet -td bash_trainingData.joblib
python3 tester.py -rn bash_ResNet -td bash_testingData.joblib -n bash_ResNet_Model.pth

