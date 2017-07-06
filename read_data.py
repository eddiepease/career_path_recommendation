import os
import glob
import h5py
import time
import json
import numpy as np
import pandas as pd
import pickle

from job_title_normalizer.ad_parsing import JobTitleNormalizer


class CVJobNormalizer():
    def __init__(self):

        # read in necessary files
        self.fnoun_plural = pickle.load(open("job_title_normalizer/data/fnoun_plural_dict.pkl", "rb"), encoding='latin1')
        self.fnoun_set = pickle.load(open("job_title_normalizer/data/fnoun_set.pkl", "rb"), encoding='latin1')
        self.spellchecker = pickle.load(open("job_title_normalizer/data/spellchecker_dict.pkl", "rb"), encoding='latin1')
        self.stopwords = pickle.load(open("job_title_normalizer/data/stopwords.pkl", "rb"), encoding='latin1')
        self.title = pickle.load(open("job_title_normalizer/data/title_dict.pkl", "rb"), encoding='latin1')
        self.token_sub = pickle.load(open("job_title_normalizer/data/token_sub_dict.pkl", "rb"), encoding='latin1')
        self.us_uk_spellchecker = pickle.load(open("job_title_normalizer/data/us_uk_spellchecker_dict.pkl", "rb"),
                                         encoding='latin1')

        # normalizer
        self.job_title_normalizer = JobTitleNormalizer(self.stopwords, self.us_uk_spellchecker, self.spellchecker,
                                                       self.fnoun_plural, self.title, self.token_sub, self.fnoun_set)

    def normalized_job(self, df, n_row, job_num=0):
        if isinstance(df['employment_history'][n_row],list):
            if len(df['employment_history'][n_row]) > 0:
                try:
                    raw_title = df['employment_history'][n_row][job_num]['raw_job_title']
                    normalized_title = self.job_title_normalizer.process(raw_title)['title_norm']
                    return normalized_title
                except KeyError:
                    pass
                except IndexError:
                    pass


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


def read_json_data(num_file,folder):

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

def save_processed_dfs(save_name):

    # define hdf5
    store = pd.HDFStore('data/cvs_v2_processed/' + save_name + '.h5')
    files = os.listdir('data/cvs_v2/').remove('_SUCCESS') # TODO: remove SUCCESS from this

    t0 = time.time()
    print('t0:',t0)

    # loop through all files
    for i in range(0,len(files)):
        print(i)
        print(time.time())

        # import and definitions
        df = read_json_data(i,folder='data/cvs_v2/') # read in df
        cv_job_normalizer = CVJobNormalizer()
        valid_indices = []
        job_feat_list = []
        job_label_list = []
        job_to_predict = 0

        for i in range(0, len(df)):
            # print(i)
            normalized_title_feat = cv_job_normalizer.normalized_job(df, n_row=i, job_num=job_to_predict + 1)
            normalized_title_label = cv_job_normalizer.normalized_job(df, n_row=i, job_num=job_to_predict)

            # track index of valid CVs
            if normalized_title_feat is not None and normalized_title_label is not None:
                valid_indices.append(i)
                job_feat_list.append(normalized_title_feat)
                job_label_list.append(normalized_title_label)

        # reduce df to only valid cols
        df_trans = df.iloc[valid_indices, :].reset_index(drop=True)
        df_trans['normalised_title_feat'] = job_feat_list
        df_trans['normalised_title_label'] = job_label_list

        # save df as HDF5
        key = str(i)
        store[key] = df_trans

    t1 = time.time()
    print('Total time taken:', t1-t0)

    # test_1_df = store['1']
    # print(test_1_df.shape)
    # test_2_df = store['2']
    # print(test_2_df.shape)

# function to read in the h5
def read_h5_files(file_name, num_files):

    df_result = pd.DataFrame()
    filename = 'data/cvs_v2_processed/' + file_name + '.h5'
    f = h5py.File(filename, 'r')
    keys = [key for key in f.keys()]

    for i in range(0,num_files):
        key = keys[i]
        df = pd.read_hdf('data/cvs_v2_processed/' + file_name + '.h5', key)
        df_result = pd.concat([df_result, df]).reset_index(drop=True)

    return df_result





if __name__ == "__main__":

    # start time
    t0 = time.time()

    file_name = 'h5_cvs'
    df = read_h5_files(file_name,num_files=1)

    print(df.shape)

    # filename = 'data/cvs_v2_processed/test_file.h5'
    # f = h5py.File(filename, 'r')
    # keys = [key for key in f.keys()]
    # print(keys)
    # # ['31172', '31607', '31635', '31642', '36334', '36541', '36718', '36852', '57293', '57486', '57568', '62617','62666', '67734', '67747', '68015', '68355', '78549']

    # List all groups
    # print("Keys: %s" % f.keys())
    # a_group_key = f.keys()[0]






    # save_processed_dfs('test_file')
    # df = pd.read_hdf('data/cvs_v2_processed/h5_cvs.h5','0')
    # store = pd.HDFStore('data/cvs_v2_processed/h5_cvs.h5')
    # print(df.shape)



    # # stuff
    # # df = read_json_data('data/cvs/')
    # # skills_profile_to_dict(save_location='data/ontology/skill-profiles/skill_profile_dict.pkl')
    # test_dict = read_ontology_data('skill-profiles', file_type='pkl')
    # # print(test_dict)
    #
    # # pickle.dump(df, open('data/all_csv.pkl','wb'))
    # #
    # #
    # #
    # #
    # # # end time
    # # t1 = time.time()
    # #
    # # duration = round(t1 - t0, 2)
    # # print('Duration is', duration, 'seconds')


