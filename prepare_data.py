import os
import re
import h5py
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from read_data import read_single_json_data,read_ontology_data,read_embeddings_json,read_h5_files_nemo,read_all_json_data
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


def process_education_history(education_history, uni_dict):

    # setup
    # education_features = np.zeros(7) # 0: top 30 uni, 1: 30-150, 2: 150-500, 3: unranked, 4:no university, 5:MBA,6:PhD
    education_features = np.zeros(504)

    if len(education_history) > 0:

        # determine the number of degrees
        patt = r"(^(master|doctor)'?s?$|(master|doctor)'?s?\s(of|in)|degree|l\.?l\.?b\.?|\bdr\b|bachelor'?s?|\b(b\.?a\.?|mb?\.?a\.?|[mb]\.?s\.?c\.?|b\.?s\.?|hons|p\.?h\.?d\.?|[bm]\.?e\.?n\.?g\.?)\b)"  # must match this
        notpatt = r"(foundation|\baa\b|\bas\b)"  # must not match this
        degrees = []
        for edu in education_history:
            if 'qualification_type' in edu and edu["qualification_type"] is not None:
                if re.search(patt, edu["qualification_type"], re.IGNORECASE) and not re.search(notpatt,
                                                                                                edu["qualification_type"],
                                                                                                re.IGNORECASE):
                    degrees.append(edu)


            if 'institution' in edu and edu["institution"] is not None:
                if re.search(r'\b(university|polytechnic)\b', edu["institution"], re.IGNORECASE):
                    degrees.append(edu)

        # fill out the features
        if len(degrees) > 0:
            for degree in degrees:
                # uni_idx = 5000 # initialise at high value

                # initiliase to guard against keyerrors
                qual_true = False
                inst_true = False
                if 'qualification_type' in degree: qual_true = True
                if 'institution' in degree: inst_true = True

                if inst_true and degree["institution"] is not None:
                    try:
                        # insert 1 in the university rank
                        cleaned_institution_name = strip_education_str(degree["institution"])
                        rank = uni_dict[cleaned_institution_name]
                        education_features[rank - 1] = 1 # 1 in university rank
                        # if rank <= 30:
                        #     education_features[0] = 1 # top 30 uni
                        # elif rank > 30 and rank <= 150:
                        #     education_features[1] = 1 # 30-150 uni
                        # elif rank > 150:
                        #     education_features[2] = 1 # 150-500 uni
                        # else:
                        #     education_features[3] = 1 # unranked
                    except KeyError:
                        # education_features[3] = 1 # unranked
                        education_features[501] = 1

                else:
                    # education_features[3] = 1 # unranked
                    education_features[501] = 1

                # MBA/PHD terms:
                mbapatt = r"(master'?s?\s(of\s|in\s)?business\sadministration|\be?\.?m\.?b\.?a\.?\b)"
                phdpatt = r"(doctor\sof|^doctor$|doctorate|\bp\.?h\.?d\.?|d\.?phil|^dr\.?$)"
                if not inst_true or degree["institution"] is None: degree["institution"] = "NONE"
                if not qual_true or degree["qualification_type"] is None: degree["qualification_type"] = "NONE"

                if (re.search(mbapatt, degree["institution"], re.IGNORECASE) or re.search(mbapatt,
                                                                                          degree["qualification_type"],
                                                                                          re.IGNORECASE)):
                    # education_features[5] = 1  # mba
                    education_features[502] = 1
                phdpatt = r"(doctor\sof|^doctor$|doctorate|\bp\.?h\.?d\.?|d\.?phil|^dr\.?$)"
                if (re.search(phdpatt, degree["institution"], re.IGNORECASE) or re.search(phdpatt,
                                                                                          degree["qualification_type"],
                                                                                          re.IGNORECASE)):
                    # education_features[6] = 1.  # phd
                    education_features[503] = 1

    else:
        # education_features[4] = 1 # no university
        education_features[500] = 1


    return education_features




#################
# save transformed CVs
####################

# function which transforms the cvs for the baseline model
# the transformed data is saved into h5 storage
def save_processed_dfs_baseline(save_name):

    # define hdf5
    store = pd.HDFStore('data/cvs_v3_baseline_processed/' + save_name + '.h5')
    cv_directory = 'data/cvs_v3/'
    files = [file for file in os.listdir(cv_directory) if file != '_SUCCESS']
    len_df = 0
    len_valid_df = 0

    t0 = time.time()
    print('t0:',t0)

    # loop through all files
    for i in range(0,len(files)):
        print(i)
        print(time.time())

        # import and definitions
        df = read_single_json_data(i,folder=cv_directory) # read in df
        valid_indices = []
        job_feat_list = []
        job_label_list = []
        len_df += len(df)

        for j in range(0, len(df)):
            norm_emp_list = df['employment_history_norm'][j]
            normalized_title_feat = None
            normalized_title_label = None
            if isinstance(norm_emp_list, list) and len(norm_emp_list) > 0:
                try: normalized_title_feat = norm_emp_list[1]['title_norm']
                except KeyError: pass
                except IndexError: pass
                try: normalized_title_label = norm_emp_list[0]['title_norm']
                except KeyError: pass

            # track index of valid CVs
            if normalized_title_feat is not None and normalized_title_label is not None:
                valid_indices.append(j)
                job_feat_list.append(normalized_title_feat)
                job_label_list.append(normalized_title_label)

        # reduce df to only valid cols
        df_trans = df.iloc[valid_indices, :].reset_index(drop=True)
        df_trans['normalised_title_feat'] = job_feat_list
        df_trans['normalised_title_label'] = job_label_list
        len_valid_df += len(df_trans)

        # save df as HDF5
        key = 'file_' + str(i)
        store[key] = df_trans

    t1 = time.time()
    print('Total time taken:', t1-t0)
    print('Dataframe length: ', len_df)
    print('Df length after preprocessing:', len_valid_df)


# function which transforms the cvs for the nemo model
# the transformed data is saved into h5 storage
def save_processed_dfs_nemo(max_roles=10):

    # define hdf5
    job_store = h5py.File('data/cvs_v4_processed/job_store.h5','w')
    seqlen_store = h5py.File('data/cvs_v4_processed/seqlen_store.h5', 'w')
    skill_store = h5py.File('data/cvs_v4_processed/skill_store.h5', 'w')
    edu_store = h5py.File('data/cvs_v4_processed/edu_store.h5', 'w')
    label_store = h5py.File('data/cvs_v4_processed/label_store.h5', 'w')
    df_store = pd.HDFStore('data/cvs_v4_processed/df_store.h5')

    files = [file for file in os.listdir('data/cvs_v4/') if file != '_SUCCESS']
    embedding_size = 100
    university_dict = read_ontology_data('universities',file_type='pkl')
    no_skill_profile_list = ['photographic developer', 'glass painter', 'tax inspector', 'laundry ironer', 'outdoor activities coordinator',
                            "special-interest groups' official", 'footwear product development manager', 'marine firefighter', 'craft shop manager',
                            'textile colourist', 'stone engraver', 'control panel tester']



    t0 = time.time()
    print('t0:', t0)

    num_positions = 0

    # job store
    # for i in range(0, 1):
    for i in range(0, len(files)):
        print(i)
        print(time.time())

        # import and definitions
        df = read_single_json_data(num_file=i,folder='data/cvs_v4/')  # read in df
        # df = read_all_json_data(folder='data/cvs_v4/')
        file_data = np.zeros(shape=(len(df),max_roles,100))
        seq_len_array = np.zeros(shape=(len(df)))
        job_embed_dict = read_ontology_data('job-word2vec',file_type='pkl')
        # last_roles = []
        last_roles = ['manager'] * len(df)
        last_roles_idx = []
        complete_roles_idx = []
        # delete_list = ['manager','consultant','advisor','engineer']

        # loop through rows
        for j in range(0,len(df)):
            person_emp_list = df['employment_history_norm'][j]
            complete_roles = 0
            if isinstance(person_emp_list, list):
                if len(person_emp_list) > 0:
                    # append sequence length
                    seq_len_array[j,] = len(person_emp_list)
                    if len(person_emp_list) == 0:
                        print('BLANK!!!')
                    # loop through roles
                    for k in range(0,len(person_emp_list)):
                        # set idx according length of employment
                        if len(person_emp_list) >= max_roles:
                            idx = max_roles - k - 1
                        else:
                            idx = len(person_emp_list) - k - 1
                        # append last role
                        if k == 0:
                            if 'title_norm' in person_emp_list[k]:
                                # last_roles.append(person_emp_list[k]['title_norm'])
                                # last_roles_idx.append(j)
                                last_roles[j] = person_emp_list[k]['title_norm']
                            else:
                                continue
                        # numpy array
                        if 'title_norm' in person_emp_list[k]:
                            norm_title = person_emp_list[k]['title_norm']
                            if norm_title not in no_skill_profile_list:
                                file_data[j,idx,:] = job_embed_dict[norm_title]

                                # # delete list include
                                # if norm_title not in delete_list:
                                complete_roles += 1
                        # to make sure we don't go over numpy idx array
                        if k >= max_roles:
                            break

                #complete roles idx
                if complete_roles == len(person_emp_list):
                    complete_roles_idx.append(j)
                    num_positions += complete_roles

        key = 'file_' + str(i)
        # key = 'file_1'

        # labels
        bm = BaselineModel(file_data,file_data)
        _,_, job_dict,_ = bm.prepare_feature_generation()
        # label_array = np.array([job_dict[job] for job in last_roles])
        ###### TEMP SOLUTION ######
        label_array = []
        for job in last_roles:
            try:
                label_array.append(job_dict[job])
            except KeyError:
                label_array.append(0)
        label_array = np.array(label_array)
        print('Label array shape is: ',label_array.shape)
        print('File data shape is: ', file_data.shape)

        # skills
        X_skill_list = []
        file_name = 'data/ontology/skill-word2vec-json/part-00000-f545a814-9c2f-420f-a022-2dd3fc62c30b.json'
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
        edu_array = np.zeros(shape=(len(df),504))
        for l in range(0, len(df)):
            # print(l)
            person_edu_list = df['education_history'][l]
            edu_array[l,:] = process_education_history(person_edu_list,university_dict)

        # # test education
        # print('start education...')
        # uni_dict = read_ontology_data('universities',file_type='pkl')
        # df_edu = pd.DataFrame(index=list(range(len(df))),columns=['cleaned_cv_name','ranking'])
        # for l in complete_roles_idx:
        #     person_edu_list = df['education_history'][l]
        #     cleaned_cv_name = 'BLANK'
        #     if len(person_edu_list) > 0:
        #         cleaned_cv_name = process_education_history(person_edu_list)
        #     df_edu.loc[l,'cleaned_cv_name'] = cleaned_cv_name
        #     try:
        #         df_edu.loc[l, 'ranking'] = uni_dict[cleaned_cv_name]
        #     except KeyError:
        #         df_edu.loc[l, 'ranking'] = 'BLANK'
        #
        # # filter + save df
        # df_edu = df_edu[df_edu['cleaned_cv_name'] != 'BLANK']
        # # df_edu = df_edu[df_edu['ranking'] == 'BLANK']
        #
        # df_main_edu = pd.concat([df_main_edu, df_edu]).reset_index(drop=True)



        # slice numpy arrays
        file_data = file_data[complete_roles_idx,:,:]
        X_skill = X_skill[complete_roles_idx,:]
        edu_array = edu_array[complete_roles_idx,:]
        seq_len_array = seq_len_array[complete_roles_idx,]
        label_array = label_array[complete_roles_idx,]
        df_array = df.iloc[complete_roles_idx,df.columns.get_loc("employment_history_norm")].reset_index(drop=True)

        print(file_data.shape)

        # save
        skill_store.create_dataset(key,data=X_skill)
        edu_store.create_dataset(key,data=edu_array)
        job_store.create_dataset(key, data=file_data)
        seqlen_store.create_dataset(key, data=seq_len_array)
        label_store.create_dataset(key, data=label_array)
        df_store[key] = df_array

    skill_store.close()
    edu_store.close()
    job_store.close()
    seqlen_store.close()
    label_store.close()

    print('Number of positions:',num_positions)

    # # education test
    # df_final_1 = df_main_edu[df_main_edu['ranking'] == 'BLANK']
    # df_final_1 = df_final_1.groupby('cleaned_cv_name').count()
    # df_final_1.to_csv('edu_no_rank.csv')
    #
    # df_final_2 = df_main_edu[df_main_edu['ranking'] != 'BLANK']
    # df_final_2 = df_final_2.groupby('cleaned_cv_name').count()
    # df_final_2.to_csv('edu_rank.csv')
    # # df_final.sort_values('cleaned_cv_name',ascending=False,inplace=True)

    # time
    t1 = time.time()
    print('Total time taken:', t1 - t0)


if __name__ == "__main__":

    # save_processed_dfs_baseline(save_name='df_store')
    save_processed_dfs_nemo(max_roles=20)