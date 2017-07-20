import os
import h5py
import time
import pickle
import numpy as np
import pandas as pd

from read_data import read_single_json_data,read_ontology_data
from job_title_normalizer.ad_parsing import JobTitleNormalizer

###############
# helper objects/functions
###############

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


#################
# save transformed CVs
####################

# function which transforms the cvs for the baseline model
# the transformed data is saved into h5 storage
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


# function which transforms the cvs for the nemo model
# the transformed data is saved into h5 storage
# TODO: alter this file to save skills embedding too.
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


if __name__ == "__main__":

    save_processed_dfs_nemo(max_roles=10)