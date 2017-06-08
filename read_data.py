import os
import time
import json
import pandas as pd

# TODO: seems like this can be made more efficient
def read_json_data():

    # define variables
    data = []
    folder = 'data/cvs/'
    files = os.listdir(folder)[:2] # TODO: change this

    # loop through files
    for file in files:
        path = folder + file
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":

    # start time
    t0 = time.time()

    df = read_json_data()
    # print(df['region'].head(50))

    # end time
    t1 = time.time()

    duration = round(t1 - t0, 2)
    print('Duration is', duration, 'seconds')


