import os
import glob
import time
import json
import numpy as np
import pandas as pd
import pickle


# TODO: any other preprocessing required?
def preprocessing(df):
    # split into 2 data frames
    df_1 = df[df['email_address'].isnull() == False]
    df_2 = df[df['email_address'].isnull() == True]

    # transform df_1
    df_1.sort_values('meta_date_of_cv_upload',ascending=False,inplace=True)
    df_1.drop_duplicates('email_address',inplace=True)

    # merge back together
    df_result = pd.concat([df_1,df_2]).reset_index(drop=True)

    return df_result


def read_json_data():

    # define variables
    data = []
    folder = 'data/cvs/'
    files = os.listdir(folder)

    # loop through files
    for file in files:
        path = folder + file
        print(file)
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))


    df = preprocessing(pd.DataFrame(data))

    return df

def read_ontology_data(ontology_name):
    folder_name = 'data/ontology/' + ontology_name
    allFiles = glob.glob(folder_name + "/*.csv")

    for file in allFiles:
        df = pd.read_csv(file)
    return df

def read_general_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def read_embeddings_json(file_path):
    skill_embeddings = {}
    with open(file_path) as f:
        for line in f:
            emb = json.loads(line)
            skill_embeddings[emb["word"]] = np.asarray(emb["vector"]["values"], dtype=float)
            
    return skill_embeddings


if __name__ == "__main__":

    # start time
    t0 = time.time()

    # stuff
    df = read_json_data()

    pickle.dump(df, open('data/all_csv.pkl','wb'))




    # end time
    t1 = time.time()

    duration = round(t1 - t0, 2)
    print('Duration is', duration, 'seconds')


