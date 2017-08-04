import os
import re
import h5py
import time
import pickle
import numpy as np
import pandas as pd

from read_data import read_single_json_data,read_ontology_data,read_embeddings_json,CVJobNormalizer
from baseline_model import BaselineModel

##############
# education
##############

def strip_education_str(string):
    """
    1.Set everything to lower case
    2.Remove stop words (respecting word boundaries)
      and replace with single space: "university" "the" "of" "at" "and" “&”
    3.Remove stop punctuation and replace with single space: “-” “,” “.”
    4.Remove single apostrophe ‘ and collapse (ie not replace with space)
    5.Trim any leading or trailing spaces, and convert any multiple spaces to single space
    """
    assert(isinstance(string,str))
    string = string.lower()
    string = re.sub(r"\b(university|the|of|at|and|&)\b", ' ', string)  # remove stop words
    string = re.sub(r"[-,.]", ' ', string)  # remove stop punctuation
    string = re.sub(r"[']", '', string)  # remove apostrophes
    string = re.sub('\s+', ' ', string).strip()  # trim incorrect whitespaces
    return string

# TODO: finish completing the KeyError changes
def process_education_history(education_history, universities):
    # print ""
    # print ""

    education_features = np.zeros(6)  # uni_A,uni_B,uni_C,uni_D,MBA,PhD

    patt = r"(^(master|doctor)'?s?$|(master|doctor)'?s?\s(of|in)|degree|l\.?l\.?b\.?|\bdr\b|bachelor'?s?|\b(b\.?a\.?|mb?\.?a\.?|[mb]\.?s\.?c\.?|b\.?s\.?|hons|p\.?h\.?d\.?|[bm]\.?e\.?n\.?g\.?)\b)"  # must match this
    notpatt = r"(foundation|\baa\b|\bas\b)"  # must not match this
    degrees = []
    for edu in education_history:
        try:
            if edu["qualification_type"] is not None:
                if re.search(patt, edu["qualification_type"], re.IGNORECASE) and not re.search(notpatt,
                                                                                               edu["qualification_type"],
                                                                                               re.IGNORECASE):
                    degrees.append(edu)
                    continue
        except KeyError:
            pass

        try:
            if edu["institution"] is not None:
                if re.search(r'\b(university|polytechnic)\b', edu["institution"], re.IGNORECASE):
                    degrees.append(edu)
        except KeyError:
            pass

    # build features from degrees list:
    # Check_1: is 'name' a substring of cleaned institution name?
    # Check_2: is 'alt_name' a substring of cleaned institution name?
    # Check_3: is 'not_name' a substring of cleaned institution name?
    # match if (check_1 OR check_2) and NOT check_3 are satisfied.

    if len(degrees) > 0:
        lowest_rank = 3  # initialises at uni_D (the worst rank)
        for degree in degrees:
            if degree["institution"] is None:
                match = False
            else:
                cleaned_institution_name = strip_education_str(degree["institution"])
                for ind, uni in enumerate(universities):
                    check1 = True if uni[0].decode("utf-8") and re.search(r"\b" + uni[0].decode("utf-8") + r"\b",
                                                                          cleaned_institution_name,
                                                                          re.IGNORECASE) else False  # name
                    check2 = True if uni[1].decode("utf-8") and re.search(r"\b" + uni[1].decode("utf-8") + r"\b",
                                                                          cleaned_institution_name,
                                                                          re.IGNORECASE) else False  # alt name
                    check3 = True if uni[2].decode("utf-8") and re.search(r"\b" + uni[2].decode("utf-8") + r"\b",
                                                                          cleaned_institution_name,
                                                                          re.IGNORECASE) else False  # not name
                    match = True if ((check1 or check2) and not check3) else False
                    if match == True:
                        break

            # calc rank and assign feature vals
            if match == False:
                rank = 3  # uni_D
            else:
                ranking = universities[ind][3]  # rank
                if type(ranking) == int:
                    if ranking <= 30:
                        rank = 0  # uni_A
                    else:
                        rank = 1  # uni_B
                else:
                    if ranking == "101-150":
                        rank = 1  # uni_B
                    else:
                        rank = 2  # uni_C

            if rank < lowest_rank: lowest_rank = rank

            # MBA/PHD terms:
            mbapatt = r"(master'?s?\s(of\s|in\s)?business\sadministration|\be?\.?m\.?b\.?a\.?\b)"
            phdpatt = r"(doctor\sof|^doctor$|doctorate|\bp\.?h\.?d\.?|d\.?phil|^dr\.?$)"
            if degree["institution"] is None: degree["institution"] = "NONE"
            if degree["qualification_type"] is None: degree["qualification_type"] = "NONE"

            if (re.search(mbapatt, degree["institution"], re.IGNORECASE) or re.search(mbapatt,
                                                                                      degree["qualification_type"],
                                                                                      re.IGNORECASE)):
                education_features[4] = 1.  # mba
            phdpatt = r"(doctor\sof|^doctor$|doctorate|\bp\.?h\.?d\.?|d\.?phil|^dr\.?$)"
            if (re.search(phdpatt, degree["institution"], re.IGNORECASE) or re.search(phdpatt,
                                                                                      degree["qualification_type"],
                                                                                      re.IGNORECASE)):
                education_features[5] = 1.  # phd
        education_features[lowest_rank] = 1.

    return education_features




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
    job_store = h5py.File('data/cvs_v3_processed/job_store.h5','w')
    seqlen_store = h5py.File('data/cvs_v3_processed/seqlen_store.h5', 'w')
    skill_store = h5py.File('data/cvs_v3_processed/skill_store.h5', 'w')
    edu_store = h5py.File('data/cvs_v3_processed/edu_store.h5', 'w')
    label_store = h5py.File('data/cvs_v3_processed/label_store.h5', 'w')

    files = [file for file in os.listdir('data/cvs_v3/') if file != '_SUCCESS']
    embedding_size = 100

    t0 = time.time()
    print('t0:', t0)

    # job store
    for i in range(0, 1):
    # for i in range(0, len(files)):
        print(i)
        print(time.time())

        # import and definitions
        df = read_single_json_data(num_file=i,folder='data/cvs_v3/')  # read in df
        file_data = np.zeros(shape=(len(df),max_roles,100))
        seq_len_array = np.zeros(shape=(len(df)))
        job_embed_dict = read_ontology_data('job-word2vec',file_type='pkl')
        last_roles = []
        last_roles_idx = []

        # loop through rows
        for j in range(0,len(df)):
            person_emp_list = df['employment_history_norm'][j]
            if isinstance(person_emp_list, list):
                if len(person_emp_list) > 0:
                    # append sequence length
                    seq_len_array[j,] = len(person_emp_list)
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

        # labels
        bm = BaselineModel(file_data,file_data)
        _,_, job_dict,_ = bm.prepare_feature_generation()
        label_array = np.array([job_dict[job] for job in last_roles])

        # skills
        X_skill_list = []
        file_name = 'data/ontology/skill-word2vec/data/skill_embeddings.json'
        skill_embeddings_dict = read_embeddings_json(file_name)

        for person in list(df['skills']):
            individual_skills = []
            for skill in person:
                try:
                    individual_skills.append(skill_embeddings_dict[skill])
                except KeyError:
                    pass

            if len(individual_skills) > 0:
                max_pool_skill = np.max(np.array(individual_skills), axis=0)  # max pooling operation
            else:
                max_pool_skill = np.zeros(shape=(embedding_size,))
            X_skill_list.append(max_pool_skill)

        X_skill = np.array(X_skill_list)

        # education
        print('Doing education...')
        unis = read_ontology_data('universities')
        edu_array = np.zeros(shape=(len(df),6))
        for l in range(0, len(df)):
            print(l)
            person_edu_list = df['education_history'][l]
            if len(person_edu_list) > 0:
                edu_array[l,:] = process_education_history(person_edu_list,unis)

        # slice numpy arrays
        file_data = file_data[last_roles_idx,:,:]
        X_skill = X_skill[last_roles_idx,:]
        edu_array = edu_array[last_roles_idx,:]
        seq_len_array = seq_len_array[last_roles_idx,]

        # save
        skill_store.create_dataset(key,data=X_skill)
        edu_store.create_dataset(key,data=edu_array)
        job_store.create_dataset(key, data=file_data)
        seqlen_store.create_dataset(key, data=seq_len_array)
        label_store.create_dataset(key, data=label_array)

    skill_store.close()
    job_store.close()
    seqlen_store.close()
    label_store.close()

    # time
    t1 = time.time()
    print('Total time taken:', t1 - t0)


if __name__ == "__main__":

    save_processed_dfs_nemo(max_roles=10)