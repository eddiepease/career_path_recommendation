import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
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
        print('preparing bag of skills features...')

        # read in pre-requisites
        skill_profile_dict, unique_skills, job_dict, reverse_job_dict = self.prepare_feature_generation()

        # create dict
        bos_dict = dict.fromkeys(unique_skills, 0)

        v = DictVectorizer(sparse=True) # TODO:save as numpy array rather than sparse matrix
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

    # create features and labels using embeddings
    def create_embedding_features(self, df, tf_idf=False, job_to_predict=0):
        print('preparing embedding features...')

        # read in pre-requisites
        _, _, job_dict, reverse_job_dict = self.prepare_feature_generation()

        features_dict = {}
        labels = []

        # read most recent job(s) from CV
        # for i in range(0,1000):
        for i in range(0, len(df)):
            normalized_title_feat = self.cv_job_normalizer.normalized_job(df, n_row=i, job_num=job_to_predict + 1)
            normalized_title_label = self.cv_job_normalizer.normalized_job(df, n_row=i, job_num=job_to_predict)

            # if no job_title in CV or job title is not in skills profile for either position
            if (normalized_title_feat is not None and normalized_title_label is not None) and \
                    (normalized_title_feat in self.ordered_job_title and normalized_title_label in self.ordered_job_title):

                job_idx = self.ordered_job_title.index(normalized_title_feat)
                features_dict[i] = self.embedding[job_idx,:]
                labels.append(job_dict[normalized_title_label])

        # convert dict into numpy array
        features = pd.DataFrame(features_dict)
        X = np.transpose(np.array(features))

        return X, labels

    # plot a histogram of distribution of frequency of job titles
    def plot_job_frequency(self, labels, save_name):
        c = Counter(labels)
        labels, values = zip(*c.items())
        plt.hist(values,bins=100)
        plt.savefig(save_name)

    # return a list of values that appear infrequently
    def return_infrequent_jobs(self, y_1, y_2, threshold):
        y = y_1 + y_2
        c = Counter(y)
        low_count_values = [key for key, value in c.items() if value < threshold]
        return low_count_values

    # method to remove jobs which don't appear often
    def remove_infrequent_jobs(self,features, labels, low_count_jobs):

        # return indices at which jobs appear that are below the threshold
        low_count_indices = [i for i,idx in enumerate(labels) if idx in low_count_jobs]
        remaining_indices = list(set(list(range(0,len(labels)))) - set(low_count_indices))

        # return the features and labels without these indices
        transformed_features = features[remaining_indices]
        transformed_labels = [labels[i] for i in remaining_indices]

        return transformed_features, transformed_labels

    # function to save data so no need to re-run expensive process
    def save_transformed_data(self, embedding, weighted, save_name):
        print('saving transformed features...')

        #generate
        if embedding:
            self.embedding, self.ordered_job_title = create_job_embedding(embedding_size=100)
            X_train, y_train = self.create_embedding_features(self.train)
            X_test, y_test = self.create_embedding_features(self.test)
        else:
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
    def load_transformed_data(self,save_name,remove_rare):

        # load data
        path = 'data/' + save_name + '/'
        self.X_train = pickle.load(open(path + "X_train.pkl", "rb"))
        self.y_train = pickle.load(open(path + "y_train.pkl", "rb"))
        self.X_test = pickle.load(open(path + "X_test.pkl", "rb"))
        self.y_test = pickle.load(open(path + "y_test.pkl", "rb"))

        print('Num train classes: ', len(set(self.y_train)))
        print('Num test classes: ', len(set(self.y_test)))

        # load data
        if remove_rare == True:
            threshold = 5
            low_freq_jobs = self.return_infrequent_jobs(self.y_train,self.y_test,threshold)
            self.X_train, self.y_train = self.remove_infrequent_jobs(self.X_train,self.y_train,
                                                                     low_count_jobs=low_freq_jobs)
            self.X_test, self.y_test = self.remove_infrequent_jobs(self.X_test, self.y_test,
                                                                   low_count_jobs=low_freq_jobs)
            print('Num train classes after remove: ', len(set(self.y_train)))
            print('Num test classes after remove: ', len(set(self.y_test)))

        print('X_train shape: ', self.X_train.shape)
        print('X_test shape: ', self.X_test.shape)

    # function to evaluate performance
    def mpr_scorer(self, clf, X, y):
        class_labels = clf.classes_
        n_class = float(len(class_labels))
        y_pred_proba = clf.predict_proba(X)
        # discard test samples for which the class was not seen in the training set
        # (typically happens when train_size <~ number of classes)
        mpr = np.mean([np.where(class_labels[y_pred_proba[i].argsort()[::-1]] == y[i])[0][0] / n_class
                       for i in range(len(y))
                       if y[i] in class_labels])
        return mpr

    # TODO: test this
    def train_and_eval_model(self, save_name):

        # load data into object
        self.load_transformed_data(save_name,remove_rare=True)

        # train model
        print('Training model...')
        t0 = time.time()
        gnb = GaussianNB()
        gnb.fit(self.X_train,self.y_train)
        t1 = time.time()
        print('Training time', t1-t0)

        # evaluate model
        print('Evaluating model...')
        mpr = self.mpr_scorer(clf=gnb,X=self.X_test,y=self.y_test)

        return mpr


if __name__ == "__main__":

    t0 = time.time()
    # create train/test set
    train, test = create_train_test_set()


    # form features
    folder_name = 'processed_sample5_unweighted_embed'
    model = BaselineModel(train,test)
    # model.save_transformed_data(embedding=True, weighted=False,save_name=folder_name)
    # model.load_transformed_data(save_name=folder_name,remove_rare=True)
    mpr = model.train_and_eval_model(folder_name)
    print('MPR: ', mpr)
