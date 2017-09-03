import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from collections import Counter

from baseline_model import BaselineModel
from read_data import read_ontology_data
from split_data import create_train_test_set_stratified_nemo
# from bn_lstm import LSTMCell,BNLSTMCell, orthogonal_initializer


class NEMO(BaselineModel):
    def __init__(self, n_files,threshold=5, restore=False):
        self.threshold = threshold
        self.X_skill_train,self.X_skill_test = create_train_test_set_stratified_nemo(data_file_name='skill_store',
                                                                                     n_files=n_files,
                                                                                     threshold=self.threshold)
        self.X_edu_train, self.X_edu_test = create_train_test_set_stratified_nemo(data_file_name='edu_store',
                                                                                      n_files=n_files,
                                                                                      threshold=self.threshold)
        self.X_job_train, self.X_job_test = create_train_test_set_stratified_nemo(data_file_name='job_store',
                                                                                      n_files=n_files,
                                                                                      threshold=self.threshold)
        self.seqlen_train, self.seqlen_test = create_train_test_set_stratified_nemo(data_file_name='seqlen_store',
                                                                                    n_files=n_files,
                                                                                    threshold=self.threshold)
        self.y_train, self.y_test = create_train_test_set_stratified_nemo(data_file_name='label_store',
                                                                                  n_files=n_files,
                                                                                  threshold=self.threshold)
        self.df_train, self.df_test = create_train_test_set_stratified_nemo(data_file_name='df_store',
                                                                            n_files=n_files,
                                                                            threshold=self.threshold)
        self.df_train = self.df_train.reset_index(drop=True)
        self.df_test = self.df_test.reset_index(drop=True)
        self.embedding_size = 100
        self.restore = restore
        BaselineModel.__init__(self, self.X_skill_train, self.X_skill_train)
        _,_,self.job_dict,self.reverse_job_dict = self.prepare_feature_generation()
        self.initialize_values()
        self.y_train = np.array([self.job_reduce_dict[job] for job in self.y_train])
        self.y_test = np.array([self.job_reduce_dict[job] for job in self.y_test])
        self.compute_graph()

    def initialize_values(self):
        self.class_labels = np.unique(np.concatenate((self.y_train, self.y_test)))
        self.n_unique_jobs = len(list(self.class_labels))
        print('Number of unique job titles: ',self.n_unique_jobs)

        # skill reduce dict
        self.job_reduce_dict = {}
        self.reverse_job_reduce_dict = {}
        for i, job in enumerate(self.class_labels):
            self.job_reduce_dict[job] = i
            self.reverse_job_reduce_dict[i] = job

        self.reduced_class_labels = np.array(range(self.n_unique_jobs))

        return self

    def generate_random_batches(self, X_skill, X_edu, X_job, X_seqlen, y, batch_size):
        idx = np.random.randint(0,len(X_job),batch_size)
        X_skill_batch = X_skill[idx,:]
        X_edu_batch = X_edu[idx,:]
        X_job_batch = X_job[idx,:,:]
        X_seqlen_batch = X_seqlen[idx,]
        y_batch = np.expand_dims(y[idx,],axis=1)
        return X_skill_batch,X_edu_batch,X_job_batch,X_seqlen_batch, y_batch

    def compute_graph(self):
        # define the compute graph with everything as 'self'
        # need to include a definition of mpr here

        # general definitions
        self.sess = tf.Session()

        self.batch_size = 1000
        self.max_roles = 20
        self.embedding_size = 100
        self.education_size = 504
        self.n_linear_hidden = self.embedding_size
        self.n_lstm_hidden = 50
        self.number_of_layers = 3

        ###########
        # encoder
        ###########

        self.max_pool_skills = tf.placeholder(dtype=tf.float32,shape=(None,self.embedding_size))
        # self.education = tf.placeholder(dtype=tf.float32,shape=(None,self.education_size))
        # add university perhaps in the future + location

        # one layer NN
        with tf.variable_scope("encoder"):
            # self.concat_rep = tf.concat([self.max_pool_skills,self.education],axis=1)
            self.concat_rep = self.max_pool_skills
            self.W_linear = tf.get_variable("W_linear",shape=(int(self.concat_rep.get_shape()[1]),self.n_linear_hidden),
                                            initializer=tf.contrib.layers.xavier_initializer())
            self.b_linear = tf.Variable(tf.constant(0.0,shape=(self.n_linear_hidden,)))
            self.encoder_output = tf.tanh(tf.matmul(self.concat_rep,self.W_linear) + self.b_linear)


        ###########
        # decoder
        ###########

        self.job_inputs = tf.placeholder(tf.float32, shape=(None, self.max_roles - 1, self.embedding_size))
        self.seqlen = tf.placeholder(tf.int32, shape=(None,))
        self.job_true = tf.placeholder(tf.int32,shape=(None,1))

        self.encoder_output = tf.expand_dims(self.encoder_output,axis=1)
        self.encoded_job_inputs = tf.concat([self.encoder_output,self.job_inputs],axis=1)
        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.n_lstm_hidden, state_is_tuple=True)
        # self.lstm = BNLSTMCell(self.n_lstm_hidden,self.training)
        # self.lstm = tf.contrib.rnn.GRUCell(self.n_lstm_hidden)
        # self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells=[self.lstm for _ in range(self.number_of_layers)],state_is_tuple=True)

        with tf.variable_scope("decoder",initializer=tf.contrib.layers.xavier_initializer()):
            self.job_outputs, self.last_states = tf.nn.dynamic_rnn(self.lstm, self.encoded_job_inputs,
                                                    sequence_length=self.seqlen,
                                                    dtype=tf.float32)
                                                    # initial_state=self.lstm.zero_state(tf.shape(self.job_inputs)[0], tf.float32))

        # output
        # self.final_job_output = tf.gather_nd(self.job_outputs,self.seqlen)
        self.actual_batch_size = tf.shape(self.job_inputs)[0]
        self.final_job_output = tf.gather_nd(self.job_outputs, tf.stack([tf.range(self.actual_batch_size), self.seqlen - 1], axis=1))

        self.W_output = tf.get_variable("W_output",shape=(self.n_lstm_hidden, self.n_unique_jobs),
                                        initializer=tf.contrib.layers.xavier_initializer())
                                        # initializer=orthogonal_initializer())
        self.b_output = tf.Variable(tf.constant(0.0, shape=(self.n_unique_jobs,)))
        self.logits = tf.matmul(self.final_job_output,self.W_output) + self.b_output

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.job_true,axis=1),
                                                                                  logits=self.logits))
        # self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        # gradient capping
        self.optimizer = tf.train.AdamOptimizer()
        self.gvs = self.optimizer.compute_gradients(self.loss)
        self.capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
        self.train_step = self.optimizer.apply_gradients(self.capped_gvs)

        self.test_probs = tf.nn.softmax(self.logits) # shape: [batch_size x unique_jobs]

        return self

    def nemo_mpr(self,y_pred_proba,y_true):
        mpr_list = [np.where(self.reduced_class_labels[y_pred_proba[i].argsort()[::-1]] == y_true[i])[0][0] / len(self.reduced_class_labels)
                       for i in range(len(y_true))
                       if y_true[i] in self.reduced_class_labels]
        mpr = np.mean(mpr_list)
        return mpr,mpr_list

    def run_nemo_model(self, n_iter, print_freq, model_name):

        saver = tf.train.Saver()
        folder_name = 'saved_models/' + model_name + '/'
        file_name = 'saved_model'

        # restore the model
        if self.restore:
            print('Restoring ', model_name, '...')
            saver = tf.train.import_meta_graph(folder_name + file_name + '.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(folder_name))

        # train the model
        else:

            self.sess.run(tf.global_variables_initializer())

            for iter in range(n_iter):
                X_skill_batch,X_edu_batch, X_job_batch,X_seqlen_batch, y_batch = self.generate_random_batches(self.X_skill_train,
                                                                                                 self.X_edu_train,
                                                                                                 self.X_job_train,
                                                                                                 self.seqlen_train,
                                                                                                 self.y_train,
                                                                                                 batch_size=self.batch_size)
                train_feed_dict = {self.max_pool_skills: X_skill_batch,
                                   # self.education: X_edu_batch,
                                   self.job_inputs: X_job_batch[:,:self.max_roles-1,:],
                                   self.seqlen: X_seqlen_batch,
                                   self.job_true: y_batch}
                self.sess.run([self.train_step],train_feed_dict)

                if iter % print_freq == 0:
                    test_feed_dict = {self.max_pool_skills: self.X_skill_test,
                                      # self.education: self.X_edu_test,
                                      self.job_inputs: self.X_job_test[:,:self.max_roles-1,:],
                                      self.seqlen: self.seqlen_test,
                                      self.job_true: np.expand_dims(self.y_test, axis=1)}

                    train_loss = self.sess.run(self.loss, train_feed_dict)
                    test_loss = self.sess.run(self.loss, test_feed_dict)

                    print('Train Loss at', iter, ": ", train_loss)
                    print('Test Loss:', test_loss)

            # saving model
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            saver.save(self.sess, folder_name + file_name)

        return self

    def evaluate_nemo(self):
        # evaluate relevant variables from compute graph
        print('evaluating...')
        test_feed_dict = {self.max_pool_skills: self.X_skill_test,
                          # self.education: self.X_edu_test,
                          self.job_inputs: self.X_job_test[:,:self.max_roles-1,:],
                          self.seqlen: self.seqlen_test,
                          self.job_true: np.expand_dims(self.y_test, axis=1)}
        test_loss, test_probs = self.sess.run([self.loss, self.test_probs], test_feed_dict)
        print('Test Loss: ', test_loss)

        # calculating MPR
        print('calculating MPR')
        mpr,_ = self.nemo_mpr(test_probs, self.y_test)

        return mpr

    def test_individual_examples(self,idx_list, num_pred_show = 10):

        # initialize
        jobs_index = ['job_' + str(i + 1) for i in range(self.max_roles)]
        results_index = ['prediction_' + str(i + 1) for i in range(num_pred_show)]
        whole_index = jobs_index + results_index
        df_results = pd.DataFrame(index=whole_index,columns=idx_list)

        test_feed_dict = {self.max_pool_skills: self.X_skill_test,
                          # self.education: self.X_edu_test,
                          self.job_inputs: self.X_job_test[:, :self.max_roles - 1, :],
                          self.seqlen: self.seqlen_test,
                          self.job_true: np.expand_dims(self.y_test, axis=1)}
        test_probs = self.sess.run(self.test_probs, test_feed_dict)


        # loop through df
        for idx in idx_list:
            row = self.df_test.iloc[idx,:][0]
            for i in range(len(row)):
                pos = len(row) - 1 - i
                col = 'job_' + str(pos+1)
                df_results.loc[col,idx] = row[i]['title_norm']

            # prediction
            prediction_idxes = test_probs[idx].argsort()[::-1][:num_pred_show]
            # prediction_idx = np.argmax(test_probs,axis=1)[idx]
            prediction_titles = [self.reverse_job_dict[self.reverse_job_reduce_dict[prediction_idxes[j]]] for j in range(num_pred_show)]
            for k in range(num_pred_show):
                col = 'prediction_' + str(k+1)
                df_results.loc[col,idx] = prediction_titles[k]

        return df_results

    def plot_error_analysis(self):

        print('Plotting performance analysis...')

        # calculate mpr list
        test_feed_dict = {self.max_pool_skills: self.X_skill_test,
                          # self.education: self.X_edu_test,
                          self.job_inputs: self.X_job_test[:, :self.max_roles - 1, :],
                          self.seqlen: self.seqlen_test,
                          self.job_true: np.expand_dims(self.y_test, axis=1)}
        test_probs = self.sess.run(self.test_probs, test_feed_dict)
        _, mpr_list = self.nemo_mpr(test_probs, self.y_test)

        ###################
        # mpr against frequency of title
        ###################

        # frequency df
        c = Counter(self.y_test)
        array_freq = np.zeros(shape=(len(test_probs),3))
        array_freq[:,0] = self.y_test
        freq_list = [c[array_freq[i,0]] for i in range(len(test_probs))]
        array_freq[:,1] = freq_list
        array_freq[:,2] = mpr_list
        df_freq = pd.DataFrame(array_freq,columns=['job_idx','count','mpr'])

        # aggregate
        bins = [0,50,100,500,2000,10000000]
        group_names = ['<50', '50-100', '100-500', '500-2000', '>2000']
        df_freq['categories'] = pd.cut(df_freq['count'], bins, labels=group_names)
        df_freq = df_freq.groupby('categories').agg({'mpr':'mean'}).reset_index()
        df_freq.to_csv('no_context_results/df_freq_edu2_context.csv')

        # # plot
        # sns.barplot(x='categories',y='mpr',data=df_freq)
        # plt.ylabel('MPR')
        # plt.title('MPR Performance against Popularity of Title')
        # plt.savefig('figures/nemo/final_skill_title_freq_error.png')



        ##################
        # mpr against number of roles
        #################

        plt.clf()

        # aggregate
        array_roles = np.transpose(np.array([self.y_test,self.seqlen_test,mpr_list]))
        df_roles = pd.DataFrame(array_roles,columns=['job_idx','num_roles','mpr'])
        df_roles = df_roles.groupby('num_roles').agg({'mpr': 'mean'}).reset_index()
        df_roles = df_roles[(df_roles['num_roles'] > 0) & (df_roles['num_roles'] <=20)]
        df_roles.to_csv('no_context_results/df_roles_edu2_context.csv')

        # # plot
        # sns.set_style("darkgrid")
        # plt.plot(df_roles['num_roles'], df_roles['mpr'],linestyle='-',marker='o',color='b')
        # # axes = plt.gca()
        # # axes.set_ylim([0, 5])
        # plt.xlabel('Number of roles')
        # plt.ylabel('MPR')
        # plt.title('MPR Performance against Experience')
        # plt.savefig('figures/nemo/final_skill_num_roles_error.png')

        return self




if __name__ == "__main__":

    model = NEMO(n_files=20,threshold=5,restore=True)
    model.run_nemo_model(n_iter=25000,print_freq=2000,model_name='final_skill_5thres')
    mpr = model.evaluate_nemo()
    print('MPR:',mpr)

    # test print individual examples
    test_list = list(range(2000,2100))
    df_test = model.test_individual_examples(idx_list=test_list,num_pred_show=20)
    print(df_test)
    df_test.to_csv('no_context_results/final_skill_indiv_2.csv')


    # test agg graph stuff
    # model.plot_error_analysis()


    # # test mpr
    # test_array = np.random.rand(10,100)
    # y_true = np.random.randint(0,100,size=(10,))
    # class_labels = np.array(range(100))
    #
    # mpr = model.nemo_mpr(test_array,y_true,class_labels)
    # print('MPR is: ', mpr)

    # test gather_nd
