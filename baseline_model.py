import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from read_data import read_ontology_data,read_embeddings_json
from split_data import create_train_test_set, create_train_test_set_stratified_baseline
from embeddings.job_embedding import create_job_embedding,create_cv_skill_embeddings


# class for a baseline model
class BaselineModel():

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def prepare_feature_generation(self):
        skills_profile_dict = read_ontology_data('skill-profiles',file_type='pkl')
        unique_job_titles = list(np.sort(list(skills_profile_dict.keys())))

        skills_dict = read_ontology_data('skill-pt',file_type='pkl')

        job_dict = {}
        for job in unique_job_titles:
            job_dict[job] = len(job_dict)

        reverse_job_dict = dict(zip(job_dict.values(), job_dict.keys()))

        return skills_profile_dict, skills_dict, job_dict, reverse_job_dict

    # create the features + labels for ML
    def create_bag_of_skills_features(self, df, include_cv_skills, tf_idf = True):
        print('preparing bag of skills features...')

        # read in pre-requisites
        skill_profile_dict, skills_dict, job_dict, reverse_job_dict = self.prepare_feature_generation()

        # define various inputs
        v = DictVectorizer(sparse=True)
        features = []
        labels = []
        feat_loc = df.columns.get_loc('normalised_title_feat')
        lab_loc = df.columns.get_loc('normalised_title_label')
        skill_loc = df.columns.get_loc('skills')

        # read most recent job(s) from CV
        # for i in range(0,1000):
        for i in range(0, len(df)):
            normalized_title_feat = df.iloc[i,feat_loc]
            normalized_title_label = df.iloc[i,lab_loc]
            skills = df.iloc[i, skill_loc]

            if include_cv_skills:

                # if no job_title in CV or job title is not in skills profile for either position
                if normalized_title_feat in skill_profile_dict and normalized_title_label in skill_profile_dict:

                    bos_dict = dict.fromkeys(list(skills_dict.keys()), 0)
                    for j, skill in enumerate(skills):
                        try:
                            bos_dict[skill] += skills_dict[skill]
                        except KeyError:
                            # print('Missing skill is: ', skill)
                            pass

                    features.append(bos_dict)

                    # create labels
                    labels.append(job_dict[normalized_title_label])

            else:

                # if no job_title in CV or job title is not in skills profile for either position
                if normalized_title_feat in skill_profile_dict and normalized_title_label in skill_profile_dict:

                    bos_dict = dict.fromkeys(list(skills_dict.keys()), 0)
                    skills = skill_profile_dict[normalized_title_feat][0]
                    weights = skill_profile_dict[normalized_title_feat][1]
                    for j, skill in enumerate(skills):
                        if tf_idf == True:
                            bos_dict[skill] += weights[j]
                        else:
                            bos_dict[skill] += 1

                    # if i % 1000 == 0:
                    #     print(list(temp_dict.keys())[:10]) # testing

                    features.append(bos_dict)

                    # create labels
                    labels.append(job_dict[normalized_title_label])

        X = v.fit_transform(features).toarray()

        return X,labels

    # create features and labels using embeddings
    def create_embedding_features(self, df, include_cv_skills, tf_idf=False):
        print('preparing embedding features...')

        # previous job
        _, _, job_dict, reverse_job_dict = self.prepare_feature_generation()

        features_dict = {}
        labels = []
        feat_loc = df.columns.get_loc('normalised_title_feat')
        lab_loc = df.columns.get_loc('normalised_title_label')

        # cv skills
        file_name = 'data/ontology/skill-word2vec/data/skill_embeddings.json'
        skill_embeddings_dict = read_embeddings_json(file_name)
        skill_loc = df.columns.get_loc('skills')

        # read most recent job(s) from CV
        # for i in range(0,1000):
        for i in range(0, len(df)):
            normalized_title_feat = df.iloc[i, feat_loc]
            normalized_title_label = df.iloc[i, lab_loc]
            skills = df.iloc[i,skill_loc]

            if include_cv_skills == 'whole':
                # if no job_title in CV or job title is not in skills profile for either position
                if normalized_title_feat in self.ordered_job_title and normalized_title_label in self.ordered_job_title:

                    features_dict[i] = create_cv_skill_embeddings(skills,skill_embeddings_dict)
                    labels.append(job_dict[normalized_title_label])

            # elif include_cv_skills == 'half':
            #     # if no job_title in CV or job title is not in skills profile for either position
            #     if normalized_title_feat in self.ordered_job_title and normalized_title_label in self.ordered_job_title:
            #
            #         features_dict[i] = create_weighted_cv_skill_embeddings(skills, skill_embeddings_dict, normalized_title_feat)
            #         labels.append(job_dict[normalized_title_label])

            else:
                # if no job_title in CV or job title is not in skills profile for either position
                if normalized_title_feat in self.ordered_job_title and normalized_title_label in self.ordered_job_title:

                    job_idx = self.ordered_job_title.index(normalized_title_feat)
                    features_dict[i] = self.embedding[job_idx,:]
                    labels.append(job_dict[normalized_title_label])

        # convert dict into numpy array
        features = pd.DataFrame(features_dict)
        X = np.transpose(np.array(features))

        return X, labels

    # # plot a histogram of distribution of frequency of job titles
    # def plot_job_frequency(self, labels, save_name):
    #     c = Counter(labels)
    #     labels, values = zip(*c.items())
    #     plt.hist(values,bins=100)
    #     plt.savefig(save_name)

    # function to save data so no need to re-run expensive process
    def save_transformed_data(self, embedding, weighted, save_name):
        print('saving transformed features...')

        #generate
        if embedding:
            self.embedding, self.ordered_job_title = create_job_embedding(embedding_size=100)
            X_train, y_train = self.create_embedding_features(self.train, include_cv_skills='whole')
            X_test, y_test = self.create_embedding_features(self.test,include_cv_skills='whole')
        else:
            X_train, y_train = self.create_bag_of_skills_features(self.train, include_cv_skills=False, tf_idf=weighted)
            X_test, y_test = self.create_bag_of_skills_features(self.test, include_cv_skills=False, tf_idf=weighted)

        # save
        path = 'data/' + save_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(X_train, open(path + "X_train.pkl", "wb"))
        pickle.dump(y_train, open(path + "y_train.pkl", "wb"))
        pickle.dump(X_test, open(path + "X_test.pkl", "wb"))
        pickle.dump(y_test, open(path + "y_test.pkl", "wb"))

        # # plot - this section is temporary
        # self.plot_job_frequency(y_train, save_name=path+'y_train_histogram.png')

    # load features and labels, either from saved or generate from
    def load_transformed_data(self,save_name):

        # load data
        path = 'data/' + save_name + '/'
        self.X_train = pickle.load(open(path + "X_train.pkl", "rb"))
        self.y_train = pickle.load(open(path + "y_train.pkl", "rb"))
        self.X_test = pickle.load(open(path + "X_test.pkl", "rb"))
        self.y_test = pickle.load(open(path + "y_test.pkl", "rb"))

        print('Num train classes: ', len(set(self.y_train)))
        print('Num test classes: ', len(set(self.y_test)))

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


    # # function to create the ecoc
    # def implement_ecoc(self):
    #
    #     # define estimator
    #     clf = SVC(probability=True)
    #
    #     # fit ecoc
    #     ecoc = OutputCodeClassifier(clf,code_size=0.01)
    #     ecoc.fit(self.X_train,self.y_train)
    #
    #     # evaluate model
    #     mpr = self.mpr_scorer(ecoc,self.X_test,self.y_test)
    #
    #     return mpr

    # function to train + eval model
    def train_and_eval_model(self, model_type, save_name):

        # load data into object
        self.load_transformed_data(save_name)

        # train model
        print('Training model...')
        t0 = time.time()
        if model_type == 'gnb':
            clf = GaussianNB()
        elif model_type == 'svm':
            clf = SVC(probability=True)
        elif model_type == 'mnb':
            clf = MultinomialNB()

        clf.fit(self.X_train,self.y_train)
        t1 = time.time()
        print('Training time', t1-t0)

        # evaluate model
        print('Evaluating model...')
        mpr = self.mpr_scorer(clf,X=self.X_test,y=self.y_test)

        return mpr



if __name__ == "__main__":

    # create train/test set
    train, test = create_train_test_set_stratified_baseline(n_files=1,threshold=1)

    # run the Baseline Model
    folder = 'whole_dataset_embed_1thres'
    model = BaselineModel(train,test)
    # model.save_transformed_data(embedding=True, weighted=False,save_name=folder)
    mpr = model.train_and_eval_model(model_type='gnb', save_name=folder)
    print('MPR: ', mpr)


    # ecoc.save_transformed_data(embedding=True,weighted=False,save_name=folder)
    # ecoc.fit(save_name=folder)
    # ecoc.predict_mpr()




    # # run Markov Model
    # print('run Markov model...')
    # model = MarkovModel(train,test)
    # trans_matrix = model.create_transition_matrix()
    # print(trans_matrix.shape)
    # print(trans_matrix[:10,:10])







    #
    # # test the bag of skills
    # X_train = pickle.load(open('data/processed_1_bos_weighted/' + "X_train.pkl", "rb"))
    #
    # test_list = list(np.sum(X_train,axis=0))
    # print(test_list)
    # # under_50 = [i for i in test_list if i < 100]
    # # print(len(under_50))
    # plt.hist(test_list,bins=100)
    # plt.savefig('tf_idf_test_hist_3.png')

    # 0: 2863
    # 1: 775
    # 2: 3894
    # 3: 679
    # 4: 4826


