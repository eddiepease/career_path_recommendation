import os
import glob
import h5py
import time
import json
import numpy as np
import pandas as pd
import pickle

from job_title_normalizer.ad_parsing import JobTitleNormalizer

#################
# helper objects + functions
################

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

# function to save baseline models into h5
def save_processed_dfs_baseline(save_name):

    # define hdf5
    store = pd.HDFStore('data/cvs_v2_processed/' + save_name + '.h5')
    files = [file for file in os.listdir('data/cvs_v2/') if file != '_SUCCESS']

    t0 = time.time()
    print('t0:',t0)

    # loop through all files
    for i in range(0,1):
        print(i)
        print(time.time())

        # import and definitions
        df = read_single_json_data(i,folder='data/cvs_v2/') # read in df
        cv_job_normalizer = CVJobNormalizer()
        valid_indices = []
        job_feat_list = []
        job_label_list = []
        job_to_predict = 0

        for j in range(0, len(df)):
            # print(i)
            normalized_title_feat = cv_job_normalizer.normalized_job(df, n_row=j, job_num=job_to_predict + 1)
            normalized_title_label = cv_job_normalizer.normalized_job(df, n_row=j, job_num=job_to_predict)

            # track index of valid CVs
            if normalized_title_feat is not None and normalized_title_label is not None:
                valid_indices.append(j)
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


# function to save nemo data into h5
def save_processed_dfs_nemo(max_roles=10):

    # define hdf5
    np_store = h5py.File('data/cvs_v3_processed/np_store.h5','w')
    df_store = pd.HDFStore('data/cvs_v3_processed/df_store.h5')
    files = [file for file in os.listdir('data/cvs_v3/') if file != '_SUCCESS']
    print(files)

    t0 = time.time()
    print('t0:', t0)

    # loop through all files
    for i in range(0, 2):
    # for i in range(0, len(files)):
        print(i)
        print(time.time())

        # import and definitions
        df = read_single_json_data(num_file=i,folder='data/cvs_v3/')  # read in df
        file_data = np.zeros(shape=(len(df),max_roles,100))
        job_embed_dict = read_ontology_data('job-word2vec',file_type='pkl')
        last_roles = []
        last_roles_idx = []

        # loop through rows
        for j in range(0,len(df)):
            print(j)
            person_emp_list = df['employment_history_norm'][j]
            if isinstance(person_emp_list, list):
                if len(person_emp_list) > 0:

                    # loop through roles
                    for k in range(0,len(person_emp_list)):
                        idx = len(person_emp_list) - k - 1
                        # append last role
                        if k == 0:
                            if 'title_norm' in person_emp_list[k]:
                                last_roles.append(person_emp_list[k]['title_norm'])
                                last_roles_idx.append(j)
                            else:
                                continue
                        # numpy array
                        if 'title_norm' in person_emp_list[k]:
                            norm_title = person_emp_list[k]['title_norm']
                            file_data[j,idx,:] = job_embed_dict[norm_title]


        key = 'file_' + str(i)

        # save df
        last_roles_df = pd.DataFrame(last_roles, columns=['normalised_title_label'])
        df = df.iloc[last_roles_idx,:].reset_index(drop=True)
        extra_df = pd.concat([df['skills'], last_roles_df], axis=1)
        extra_df.to_hdf(df_store,key=key)
        # df_store.append(key, extra_df, data_columns=True)

        # save numpy array
        print(file_data.shape)
        file_data = file_data[last_roles_idx,:,:]
        print(file_data.shape)
        np_store.create_dataset(key, data=file_data)

    np_store.close()

    # time
    t1 = time.time()
    print('Total time taken:', t1 - t0)



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