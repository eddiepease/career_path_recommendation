import os
import re
import glob
import h5py
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
    df_1 = df[df['cv_email'].isnull() == False]
    df_2 = df[df['cv_email'].isnull() == True]

    # transform df_1
    df_1.sort_values('revision_date',ascending=False,inplace=True)
    df_1.drop_duplicates('cv_email',inplace=True)

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
    files = os.listdir(folder)

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
def read_h5_files_baseline(folder_name, file_name, num_files):

    df_result = pd.DataFrame()
    filename = folder_name + file_name + '.h5'
    # f = h5py.File(filename, 'r')

    for i in range(0,num_files):
        key = 'file_' + str(i)
        df = pd.read_hdf(folder_name + file_name + '.h5', key)
        df_result = pd.concat([df_result, df]).reset_index(drop=True)

    return df_result

# function to read in the h5
def read_h5_files_nemo(np_file_name, num_files):

    folder = 'data/cvs_v4_processed/'

    # prepare np
    np_fullpath = folder + np_file_name + '.h5'
    np_list = []
    np_f = h5py.File(np_fullpath, 'r')

    for i in range(0,num_files):
        key = 'file_' + str(i)
        np_list.append(np_f[key][:])

    # key = 'file_1'
    # np_list.append(np_f[key][:])

    # tidy up
    np_f.close()
    np_result = np.concatenate(np_list)

    return np_result


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


#################
# universities
###################

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


def universities_to_dict(save_location):
    df_unis = read_ontology_data('universities')
    uni_dict = {}

    for i in range(len(df_unis)):
        cleaned_uni = strip_education_str(df_unis.loc[i,'name'])
        rank = i + 1
        uni_dict[cleaned_uni] = rank
        if isinstance(df_unis.loc[i, 'alt_name'], str):
            cleaned_abrev_uni = strip_education_str(df_unis.loc[i, 'alt_name'])
            uni_dict[cleaned_abrev_uni] = rank

    # save dictionary
    pickle.dump(uni_dict, open(save_location, 'wb'))



if __name__ == "__main__":


    # save_processed_dfs_nemo()
    # np_test, df_test = read_h5_files_nemo(np_file_name='np_store',df_file_name='df_store',num_files=1)
    # print(df_test.shape)
    # print(np_test.shape)

    # skills_pt_to_dict('data/ontology_v4/')
    # skills_profile_to_dict()

    # skills_profile_to_dict('data/ontology/skill-profiles/skill_profile_dict.pkl')
    skills_pt_to_dict('data/ontology/skill-pt/skill_pt_dict.pkl')

    # universities_to_dict('data/ontology/universities/university_ranking.pkl')