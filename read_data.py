import os
import glob
import time
import json
import pandas as pd

def read_json_data():

    # define variables
    data = []
    folder = 'data/cvs/'
    files = os.listdir(folder)[:1] # TODO: change this

    # loop through files
    for file in files:
        path = folder + file
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)

    return df

def read_ontology_data(ontology_name):
    folder_name = 'data/ontology/' + ontology_name
    allFiles = glob.glob(folder_name + "/*.csv")

    for file in allFiles:
        df = pd.read_csv(file)
    return df


# TODO: complete function to remove duplicates
def preprocessing():
    return 0

if __name__ == "__main__":

    # start time
    t0 = time.time()

    df = read_json_data()
    print(df.columns)

    # end time
    t1 = time.time()

    duration = round(t1 - t0, 2)
    print('Duration is', duration, 'seconds')


