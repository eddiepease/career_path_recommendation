import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from read_data import read_h5_files_nemo
from baseline_model import BaselineModel
from split_data import create_train_test_set_stratified_nemo

# TODO: alter to take account of stratified split
class NEMO(BaselineModel):
    def __init__(self, n_files,threshold=1):
        self.threshold = threshold
        self.X_skill_train,self.X_skill_test = create_train_test_set_stratified_nemo(data_file_name='skill_store',
                                                                                     n_files=n_files,
                                                                                     threshold=self.threshold)
        self.X_job_train, self.X_job_test = create_train_test_set_stratified_nemo(data_file_name='job_store',
                                                                                      n_files=n_files,
                                                                                      threshold=self.threshold)
        self.y_train, self.y_test = create_train_test_set_stratified_nemo(data_file_name='label_store',
                                                                                  n_files=n_files,
                                                                                  threshold=self.threshold)
        self.embedding_size = 100
        BaselineModel.__init__(self, self.X_skill_train, self.X_skill_train)
        _,_,self.job_dict,self.reverse_job_dict = self.prepare_feature_generation()
        self.n_unique_jobs = len(list(self.job_dict.keys()))
        self.compute_graph()

    def generate_random_batches(self, X_skill, X_job, y, batch_size):
        idx = np.random.randint(0,len(X_job),batch_size)
        X_skill_batch = X_skill[idx,:]
        X_job_batch = X_job[idx,:,:]
        y_batch = np.expand_dims(y[idx,],axis=1)
        return X_skill_batch,X_job_batch,y_batch

    # TODO: change sampled softmax according to how many
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
            self.W_linear = tf.Variable(tf.truncated_normal(shape=(int(self.concat_rep.get_shape()[1]),self.n_linear_hidden)))
            self.b_linear = tf.Variable(tf.constant(0.1,shape=(self.n_linear_hidden,)))
            self.encoder_output = tf.tanh(tf.matmul(self.concat_rep,self.W_linear) + self.b_linear)

        ###########
        # decoder
        ###########

        self.job_inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.max_roles, self.embedding_size))
        self.job_true = tf.placeholder(tf.int32,shape=(self.batch_size,1))

        self.encoder_output = tf.expand_dims(self.encoder_output,axis=1)
        self.encoded_job_inputs = tf.concat([self.encoder_output,self.job_inputs],axis=1)
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm_hidden, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            # TODO: think about sequence length
            self.job_outputs, _ = tf.nn.dynamic_rnn(self.lstm, self.encoded_job_inputs,
                                        initial_state=self.lstm.zero_state(self.batch_size, tf.float32))

        # output
        self.final_job_output = self.job_outputs[:,self.max_roles-1,:]

        self.W_output = tf.Variable(tf.truncated_normal(shape=(self.n_lstm_hidden, self.n_unique_jobs)))
        self.b_output = tf.Variable(tf.constant(0.1, shape=(self.n_unique_jobs,)))
        self.logits = tf.matmul(self.final_job_output,self.W_output) + self.b_output

        # calculate loss
        # training
        self.softmax_size = 50
        self.W_softmax = tf.get_variable("proj_w", [self.n_unique_jobs, self.n_unique_jobs], dtype=tf.float32)
        self.b_softmax = tf.get_variable("proj_b", [self.n_unique_jobs], dtype=tf.float32)
        self.train_loss = tf.nn.sampled_softmax_loss(weights=self.W_softmax,
                                            biases=self.b_softmax,
                                            labels=self.job_true,
                                            inputs=self.logits,
                                            num_sampled=self.softmax_size,
                                            num_classes=self.n_unique_jobs,
                                            partition_strategy="div")

        self.train_loss = tf.reduce_mean(self.train_loss)

        # # evaluation
        # logits = tf.matmul(inputs, tf.transpose(weights))
        #     logits = tf.nn.bias_add(logits, biases)
        #     labels_one_hot = tf.one_hot(labels, n_classes)
        #     loss = tf.nn.softmax_cross_entropy_with_logits(
        #         labels=labels_one_hot,
        #         logits=logits)

        self.train_step = tf.train.AdamOptimizer().minimize(self.train_loss)

        # testing
        self.y_true_one_hot = tf.squeeze(tf.one_hot(self.job_true,self.n_unique_jobs),axis=1)
        self.test_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true_one_hot,logits=self.logits) # shape: [batch_size,]
        self.test_probs = tf.nn.softmax(self.logits) # shape: [batch_size x unique_jobs]

        return self

    def nemo_mpr(self,y_pred_proba,y_true,class_labels):
        mpr = np.mean([np.where(class_labels[y_pred_proba[i].argsort()[::-1]] == y_true[i])[0][0] / len(class_labels)
                       for i in range(len(y_true))
                       if y_true[i] in class_labels])
        return mpr

    def train_nemo_model(self, n_iter, print_freq):
        # train the model using a session
        self.sess.run(tf.global_variables_initializer())

        for iter in range(n_iter):
            X_skill_batch,X_job_batch,y_batch = self.generate_random_batches(self.X_skill_train,
                                                                             self.X_job_train,
                                                                             self.y_train,
                                                                             batch_size=self.batch_size)
            train_feed_dict = {self.max_pool_skills: X_skill_batch,
                               self.job_inputs: X_job_batch,
                               self.job_true: y_batch}
            self.sess.run([self.train_step],train_feed_dict)

            if iter % print_freq == 0:
                train_loss = self.sess.run(self.train_loss,train_feed_dict)
                print('Train Loss at', iter, ": ", train_loss)

    # TODO: complete this
    def evaluate(self):
        #
        pass

    # TODO: complete this
    def test_individual_examples(self):
        pass


if __name__ == "__main__":

    model = NEMO(n_files=1)
    # nemo.train_nemo_model(n_iter=1000,print_freq=100)

    # test mpr
    test_array = np.random.rand(10,100)
    y_true = np.random.randint(0,100,size=(10,))
    class_labels = np.array(range(100))

    mpr = model.nemo_mpr(test_array,y_true,class_labels)
    print('MPR is: ', mpr)