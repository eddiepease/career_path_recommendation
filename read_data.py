import os
import glob
import h5py
import json
import numpy as np
import pandas as pd
import pickle

#################
# helper objects + functions
################

def preprocessing(df):
    # split into 2 data frames
    df_1 = df[df['email_address'].isnull() == False]
    df_2 = df[df['email_address'].isnull() == True]

    # transform df_1
    df_1.sort_values('revision_date',ascending=False,inplace=True)
    df_1.drop_duplicates('email_address',inplace=True)

    # merge back together
    df_result = pd.concat([df_1,df_2]).reset_index(drop=True)

    return df_result

###############
# reading cvs
##############

def read_single_json_data(num_file,folder):

    # define variables
    data = []
    file = os.listdir(folder)[num_file]

    # loop through files
    path = folder + file
    print(file)
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))


    df = preprocessing(pd.DataFrame(data))

    return df


def read_all_json_data(folder):
    # define variables
    data = []
    files = os.listdir(folder)[:1]

    # loop through files
    for file in files:
        print(file)
        path = folder + file
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))

    df = preprocessing(pd.DataFrame(data))
    return df

# function to read in the h5
def read_h5_files_baseline(file_name, num_files):

    df_result = pd.DataFrame()
    filename = 'data/cvs_v2_processed/' + file_name + '.h5'
    f = h5py.File(filename, 'r')
    keys = [key for key in f.keys()]

    for i in range(0,num_files):
        key = keys[i]
        df = pd.read_hdf('data/cvs_v2_processed/' + file_name + '.h5', key)
        df_result = pd.concat([df_result, df]).reset_index(drop=True)

    return df_result

# function to read in the h5
def read_h5_files_nemo(np_file_name, df_file_name, num_files):

    folder = 'data/cvs_v3_processed/'

    # prepare df
    df_fullpath = folder + df_file_name + '.h5'
    df_result = pd.DataFrame()
    # df_f = h5py.File(df_filename, 'r')

    # prepare np
    np_fullpath = folder + np_file_name + '.h5'
    np_list = []
    np_f = h5py.File(np_fullpath, 'r')

    for i in range(0,num_files):
        key = 'file_' + str(i)

        # df
        df = pd.read_hdf(df_fullpath, key)
        df_result = pd.concat([df_result, df]).reset_index(drop=True)

        # np
        np_list.append(np_f[key][:])

    # tidy up
    np_f.close()
    np_result = np.concatenate(np_list)

    return np_result,df_result


#######################
# read ontologies + other data
######################

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

def skills_pt_to_dict(save_location):
    skills_dict = {}

    # read in skills df
    skills_df = read_ontology_data('skill-pt')
    skills = list(skills_df['skill'])
    weights = list(skills_df['idf_weight'])

    for i, skill in enumerate(skills):
        print(i)
        skills_dict[skill] = weights[i]

    # save dictionary
    pickle.dump(skills_dict, open(save_location, 'wb'))



if __name__ == "__main__":


    # save_processed_dfs_nemo()
    np_test, df_test = read_h5_files_nemo(np_file_name='np_store',df_file_name='df_store',num_files=1)
    print(df_test.shape)
    print(np_test.shape)