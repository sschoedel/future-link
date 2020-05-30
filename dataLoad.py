import requests
import re
from zipfile import ZipFile
import os
import reader

def getIHMEData():
    url = "https://ihmecovid19storage.blob.core.windows.net/latest/ihme-covid19.zip"

    dwnloadsFolder = "downloads/"
    if not os.path.exists(dwnloadsFolder):
        os.mkdir(dwnloadsFolder)

    def download_url(url, save_path, chunk_size=128):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    outFile = "tempZip.zip"
    download_url(url, dwnloadsFolder + outFile)

    with ZipFile(dwnloadsFolder + outFile, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        names = zipObj.namelist()
        zipObj.extractall(path = dwnloadsFolder)

    os.remove(dwnloadsFolder + "tempZip.zip")
    with os.scandir(dwnloadsFolder) as entries:
        for entry in entries:
            with os.scandir(dwnloadsFolder+ entry.name + "/") as docs:
                for doc in docs:
                    if doc.name.startswith("Summary"):
                        with open(dwnloadsFolder + entry.name + "/" + doc.name) as file:
                            masterDict = reader.readIHMEFile(file.name)
                        
    return masterDict

if __name__ == '__main__':
    print(getIHMEData())
        


