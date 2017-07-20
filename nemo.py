import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from baseline_model import BaselineModel
from split_data import create_train_test_set_stratified_nemo
from embeddings.job_embedding import read_embeddings_json


class NEMO(BaselineModel):
    def __init__(self, df_train, df_test, np_train, np_test):
        BaselineModel.__init__(self, np_train, np_test)
        self.df_train = df_train
        self.df_test = df_test
        self.np_train = np_train
        self.np_test = np_test
        self.embedding_size = 100
        _,_,self.job_dict,self.reverse_job_dict = self.prepare_feature_generation()
        self.n_unique_jobs = len(list(self.job_dict.keys()))


    def generate_random_batches(self, X_skill, X_job, y, batch_size):
        idx = np.random.randint(0,len(X_job),batch_size)
        X_skill_batch = X_skill[idx,:]
        X_job_batch = X_job[idx,:]
        y_batch = y[idx,:]
        return X_skill_batch,X_job_batch,y_batch

    # TODO: move this to prepare_data script
    def prepare_data(self, X_job, df):

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
                max_pool_skill = np.max(np.array(individual_skills),axis=0) # max pooling operation
            else:
                max_pool_skill = np.zeros(shape=(self.embedding_size,))
            X_skill_list.append(max_pool_skill)

        X_skill = np.array(X_skill_list)

        # labels
        final_job_label = list(df['normalised_title_label'])
        labels = np.array([self.job_dict[job] for job in final_job_label])

        return X_skill, X_job, labels

    # TODO: test this
    def save_transformed_data_nemo(self,save_name):
        print('saving transformed data...')

        # transform data
        X_skill_train,X_job_train, y_train = self.prepare_data(self.np_train,self.df_train)
        X_skill_test,X_job_test, y_test = self.prepare_data(self.np_test, self.df_test)

        # save data
        path = 'data/' + save_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(X_skill_train, open(path + "X_skill_train.pkl", "wb"))
        pickle.dump(X_job_train, open(path + "X_job_train.pkl", "wb"))
        pickle.dump(y_train, open(path + "y_train.pkl", "wb"))
        pickle.dump(X_skill_test, open(path + "X_skill_test.pkl", "wb"))
        pickle.dump(X_job_test, open(path + "X_job_test.pkl", "wb"))
        pickle.dump(y_test, open(path + "y_test.pkl", "wb"))

    # load features and labels, either from saved or generate from
    # TODO:check this
    def load_transformed_data_nemo(self,save_name):

        # load data
        path = 'data/' + save_name + '/'
        self.X_skill_train = pickle.load(open(path + "X_skill_train.pkl", "rb"))
        self.X_job_train = pickle.load(open(path + "X_job_train.pkl", "rb"))
        self.y_train = pickle.load(open(path + "y_train.pkl", "rb"))
        self.X_skill_test = pickle.load(open(path + "X_skill_test.pkl", "rb"))
        self.X_job_test = pickle.load(open(path + "X_job_test.pkl", "rb"))
        self.y_test = pickle.load(open(path + "y_test.pkl", "rb"))

    # TODO: complete this
    # TODO: check this
    def compute_graph(self):
        # define the compute graph with everything as 'self'
        # need to include a definition of mpr here

        # general definitions
        self.sess = tf.Session()

        self.batch_size = 64
        self.max_roles = 10
        self.embedding_size = 100
        self.n_linear_hidden = self.embedding_size
        self.n_lstm_hidden = 100

        ###########
        # encoder
        ###########

        self.max_pool_skills = tf.placeholder(dtype=tf.float32,shape=(self.batch_size,self.embedding_size))
        # add university perhaps in the future + location

        # one layer NN
        with tf.variable_scope("encoder"):
            self.concat_rep = self.max_pool_skills
            self.W_linear = tf.Variable(tf.truncated_normal(shape=(tf.shape(self.concat_rep)[1],self.n_linear_hidden)))
            self.b_linear = tf.Variable(tf.constant(0.1,shape=(self.n_linear_hidden)))
            self.encoder_output = tf.tanh(tf.matmul(self.concat_rep,self.W_linear) + self.b_linear)

        ###########
        # decoder
        ###########

        self.job_inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.max_roles - 1, self.embedding_size))
        self.job_true = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_unique_jobs))

        self.encoder_output = tf.expand_dims(self.encoder_output,axis=1)
        self.encoded_job_inputs = tf.concat([self.encoder_output,self.job_inputs],axis=1)
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm_hidden, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            # TODO: add sequence length
            self.job_outputs, _ = tf.nn.dynamic_rnn(self.lstm, self.encoded_job_inputs,
                                        initial_state=self.lstm.zero_state(self.batch_size, tf.float32))

        # output
        self.final_job_output = tf.squeeze(self.job_outputs[:,self.max_roles,:],axis=1) # does this extend beyond length of array?

        self.W_output = tf.Variable(tf.truncated_normal(shape=(self.n_lstm_hidden, self.n_unique_jobs)))
        self.b_output = tf.Variable(tf.constant(0.1, shape=(self.n_unique_jobs)))
        self.logits = tf.matmul(self.final_job_output,self.W_output) + self.b_output

        # calculate loss
        # # TODO: implement sampled softmax loss
        # if mode == "train":
        #     loss = tf.nn.sampled_softmax_loss(
        #         weights=weights,
        #         biases=biases,
        #         labels=labels,
        #         inputs=inputs,
        #         ...,
        #         partition_strategy="div")
        # elif mode == "eval":
        #     logits = tf.matmul(inputs, tf.transpose(weights))
        #     logits = tf.nn.bias_add(logits, biases)
        #     labels_one_hot = tf.one_hot(labels, n_classes)
        #     loss = tf.nn.softmax_cross_entropy_with_logits(
        #         labels=labels_one_hot,
        #         logits=logits)

        self.train_step = tf.train.AdamOptimizer().minimise(loss)



        # TODO: add mpr scorer


        pass

    def train(self, n_iter):
        # TODO: complete this
        # train the model using a session
        self.sess.run(tf.global_variables_initializer())

        for i in range(n_iter):
            X_skill_batch,X_job_batch,y_batch = self.generate_random_batches(self.X_skill_train,
                                                                             self.X_job_train,
                                                                             self.y_train,
                                                                             batch_size=self.batch_size)
            train_feed_dict = {self.max_pool_skills: X_job_batch,
                               self.job_inputs: X_job_batch,
                               self.job_true: y_batch}
            self.train_step.run(train_feed_dict)

        #TODO: print out the loss every so often




        pass

    def evaluate(self):
        #
        pass

    def test_individual_examples(self):
        pass


if __name__ == "__main__":

    folder_name = 'nemo_test'
    df_train, df_test, np_train, np_test = create_train_test_set_stratified_nemo(n_files=1)
    nemo = NEMO(df_train,df_test,np_train,np_test)
    nemo.save_transformed_data_nemo(save_name=folder_name)