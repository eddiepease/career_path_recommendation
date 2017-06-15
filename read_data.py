import os
import glob
import time
import json
import pandas as pd


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
    folder = 'data/cvs_v2/'
    files = os.listdir(folder)[1:2] # TODO: change this

    # loop through files
    for file in files:
        path = folder + file
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


if __name__ == "__main__":

    # start time
    t0 = time.time()

    # stuff
    df = read_json_data()

    print(df.head(10))
    print(df.columns)

    df_test = preprocessing(df)
    # print(len(df_test))
    # # print(df_test.head(10))
    # # print(df_test.tail(10))




    # end time
    t1 = time.time()

    duration = round(t1 - t0, 2)
    print('Duration is', duration, 'seconds')


