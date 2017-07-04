import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from collections import defaultdict,Counter

from eda import CVJobNormalizer
from read_data import read_ontology_data
from split_data import create_train_test_set
from embeddings.job_embedding import create_job_embedding


class BaselineModel():
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.cv_job_normalizer = CVJobNormalizer()
        # self.train_features = create
        # self.eval_metric = Evaluation_Metric()
        # read this embedding in

    def prepare_feature_generation(self):
        skills_profile_dict = read_ontology_data('skill-profiles',file_type='pkl')

        skills_profile_df = read_ontology_data('skill-profiles')
        unique_job_titles = list(np.sort(skills_profile_df['title'].unique()))
        unique_skills = list(np.sort(skills_profile_df['skill'].unique()))

        job_dict = {}
        for job in unique_job_titles:
            job_dict[job] = len(job_dict)

        reverse_job_dict = dict(zip(job_dict.values(), job_dict.keys()))

        return skills_profile_dict, unique_skills, job_dict, reverse_job_dict

    # create the features + labels for ML
    def create_bag_of_skills_features(self, df, tf_idf = True, job_to_predict=0):

        # read in pre-requisites
        skill_profile_dict, unique_skills, job_dict, reverse_job_dict = self.prepare_feature_generation()

        # create dict
        bos_dict = dict.fromkeys(unique_skills, 0)

        v = DictVectorizer(sparse=True)
        features = []
        labels = []

        # read most recent job(s) from CV
        # for i in range(0,1000):
        for i in range(0, len(df)):
            normalized_title_feat = self.cv_job_normalizer.normalized_job(df, n_row=i, job_num=job_to_predict + 1)
            normalized_title_label = self.cv_job_normalizer.normalized_job(df, n_row=i, job_num=job_to_predict)

            # if no job_title in CV or job title is not in skills profile for either position
            if (normalized_title_feat is not None and normalized_title_label is not None) and \
                    (normalized_title_feat in skill_profile_dict and normalized_title_label in skill_profile_dict):

                temp_dict = bos_dict
                normalized_title_feat = self.cv_job_normalizer.normalized_job(df, n_row=i, job_num=job_to_predict + 1)
                # print(df['employment_history'][i])
                # print(normalized_title_feat)
                skills = skill_profile_dict[normalized_title_feat][0]
                weights = skill_profile_dict[normalized_title_feat][1]
                for i, skill in enumerate(skills):
                    if tf_idf == True:
                        temp_dict[skill] += weights[i]
                    else:
                        temp_dict[skill] += 1

                features.append(temp_dict)

                # create labels
                labels.append(job_dict[normalized_title_label])

        X = v.fit_transform(features)

        return X,labels

    # TODO: complete this without duplication of code
    def create_embedding_features(self,embeddings_filepath):
        # run the embedding
        embedding, ordered_job_title = create_job_embedding(embedding_size=100)


        # loop through the CVs, (cutting out jobs as previously)

        # find the index of normalized job title

        # extract embedding at this index and incrementally build features + labels
        pass

    # plot a histogram of distribution of frequency of job titles
    def plot_job_frequency(self, labels, save_name):
        c = Counter(labels)
        labels, values = zip(*c.items())
        plt.hist(values,bins=100)
        plt.savefig(save_name)


    # method to remove jobs which don't appear often
    def remove_infrequent_jobs(self,features, labels, threshold):

        # return indices at which jobs appear that are below the threshold
        c = Counter(labels)
        low_count_values = [key for key,value in c.items() if value < threshold]
        low_count_indices = [i for i,idx in enumerate(labels) if idx in low_count_values]
        remaining_indices = list(set(list(range(0,len(labels)))) - set(low_count_indices))

        # return the features and labels without these indices
        transformed_features = features[remaining_indices]
        transformed_labels = [labels[i] for i in remaining_indices]

        return transformed_features, transformed_labels

    # function to save data so no need to re-run expensive process
    def save_transformed_data(self, weighted, save_name):

        #generate
        X_train, y_train = self.create_bag_of_skills_features(self.train, tf_idf=weighted)
        X_test, y_test = self.create_bag_of_skills_features(self.test, tf_idf=weighted)

        # save
        path = 'data/' + save_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(X_train, open(path + "X_train.pkl", "wb"))
        pickle.dump(y_train, open(path + "y_train.pkl", "wb"))
        pickle.dump(X_test, open(path + "X_test.pkl", "wb"))
        pickle.dump(y_test, open(path + "y_test.pkl", "wb"))

        # plot - this section is temporary
        self.plot_job_frequency(y_train, save_name=path+'y_train_histogram.png')

    # load features and labels, either from saved or generate from
    def load_transformed_data(self,save_name):

        path = 'data/' + save_name + '/'
        self.X_train = pickle.load(open(path + "X_train.pkl", "rb"))
        self.y_train = pickle.load(open(path + "y_train.pkl", "rb"))
        self.X_test = pickle.load(open(path + "X_test.pkl", "rb"))
        self.y_test = pickle.load(open(path + "y_test.pkl", "rb"))


    def train_svm(self):
        # svm = SVC()

        pass

    def train_naive_bayes(self):
        pass

    def evaluate_model(self):
        pass



    def define_and_train_model(self):

        # this needs to include some reference to the evaluation metric

        pass

if __name__ == "__main__":

    t0 = time.time()
    # create train/test set
    train, test = create_train_test_set()


    # form features
    folder_name = 'processed_sample_unweighted_bos'
    model = BaselineModel(train,test)
    model.save_transformed_data(weighted=False,save_name=folder_name)
    model.load_transformed_data(save_name=folder_name)

    print(model.X_train.shape)
    print(len(model.y_train))
    print(model.X_test.shape)
    print(len(model.y_test))




    # X_trans, y_trans = model.remove_infrequent_jobs(X,y,threshold=2)
    # print(X_trans.shape)
    # print(len(y_trans))



    # model.create_train_test()
    # X_train = model.X_train
    # y_train = model.y_train
    #
    # t1 = time.time()
    #
    # print('Duration is', t1-t0)
    #
    # print(X_train.shape)
    # print(len(y_train))