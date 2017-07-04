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


def read_json_data(folder):

    # define variables
    data = []
    files = os.listdir(folder)[:1]

    # loop through files
    for file in files:
        path = folder + file
        print(file)
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))


    df = preprocessing(pd.DataFrame(data))

    return df

def read_ontology_data(ontology_name, file_type='csv'):
    folder_name = 'data/ontology/' + ontology_name
    allFiles = glob.glob(folder_name + "/*." + file_type)

    if file_type == 'csv':
        for file in allFiles:
            output = pd.read_csv(file)
    elif file_type == 'pkl':
        for file in allFiles:
            output = pickle.load( open( file, "rb" ) )
    else:
        assert (file_type == 'csv' or file_type == 'pkl'), "File type must be either CSV or PKL!"

    assert (len(allFiles) > 0), "allFiles list can't be of length zero!"

    return output

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

# function to convert the skills profile csv to a dict
def skills_profile_to_dict(save_location):
    # define dict
    skills_profile_dict = {}
    # {job title: [[skills],[TF-IDF weight],[normalization]] sorted by weight highest to lowest

    # read in skills profiles + order + normalize TD-IDF scores
    skills_profile_df = read_ontology_data('skill-profiles')
    skills_profile_df.sort_values(['title', 'weight'], ascending=[False, False], inplace=True)
    skills_profile_df.reset_index(drop=True, inplace=True)
    skills_profile_df = skills_profile_df.assign(
        normalized=skills_profile_df['weight'].div(skills_profile_df.groupby('title')['weight'].transform('sum')))

    unique_job_titles = list(np.sort(skills_profile_df['title'].unique()))

    for i,job in enumerate(unique_job_titles):
        print(i)
        temp_df = skills_profile_df[skills_profile_df['title'] == job]
        skills_profile_dict[job] = [list(temp_df['skill']), list(temp_df['weight']), list(temp_df['normalized'])]

    # save dictionary
    pickle.dump(skills_profile_dict, open(save_location, 'wb'))



if __name__ == "__main__":

    # start time
    t0 = time.time()

    # stuff
    # df = read_json_data('data/cvs/')
    # skills_profile_to_dict(save_location='data/ontology/skill-profiles/skill_profile_dict.pkl')
    test_dict = read_ontology_data('skill-profiles', file_type='pkl')
    # print(test_dict)

    # pickle.dump(df, open('data/all_csv.pkl','wb'))
    #
    #
    #
    #
    # # end time
    # t1 = time.time()
    #
    # duration = round(t1 - t0, 2)
    # print('Duration is', duration, 'seconds')


