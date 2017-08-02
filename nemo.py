import os
import numpy as np
import tensorflow as tf

from baseline_model import BaselineModel
from split_data import create_train_test_set_stratified_nemo

# TODO: think about how to incorporate varying sequence length
# TODO: think about only dealing with number of classes available in data
# TODO: think about adding education
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

        self.max_pool_skills = tf.placeholder(dtype=tf.float32,shape=(None,self.embedding_size))
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

        self.job_inputs = tf.placeholder(tf.float32, shape=(None, self.max_roles, self.embedding_size))
        self.job_true = tf.placeholder(tf.int32,shape=(None,1))

        self.encoder_output = tf.expand_dims(self.encoder_output,axis=1)
        self.encoded_job_inputs = tf.concat([self.encoder_output,self.job_inputs],axis=1)
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm_hidden, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            self.job_outputs, _ = tf.nn.dynamic_rnn(self.lstm, self.encoded_job_inputs,
                                        initial_state=self.lstm.zero_state(tf.shape(self.job_inputs)[0], tf.float32))

        # output
        self.final_job_output = self.job_outputs[:,self.max_roles-1,:]

        self.W_output = tf.Variable(tf.truncated_normal(shape=(self.n_lstm_hidden, self.n_unique_jobs)))
        self.b_output = tf.Variable(tf.constant(0.1, shape=(self.n_unique_jobs,)))
        self.logits = tf.matmul(self.final_job_output,self.W_output) + self.b_output

        # calculate loss
        # training
        # self.softmax_size = 50
        # self.W_softmax = tf.get_variable("proj_w", [self.n_unique_jobs, self.n_unique_jobs], dtype=tf.float32)
        # self.b_softmax = tf.get_variable("proj_b", [self.n_unique_jobs], dtype=tf.float32)
        # self.train_loss = tf.nn.sampled_softmax_loss(weights=self.W_softmax,
        #                                     biases=self.b_softmax,
        #                                     labels=self.job_true,
        #                                     inputs=self.logits,
        #                                     num_sampled=self.softmax_size,
        #                                     num_classes=self.n_unique_jobs,
        #                                     partition_strategy="div")

        self.y_one_hot = tf.squeeze(tf.one_hot(self.job_true,self.n_unique_jobs),axis=1)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y_one_hot)

        self.loss = tf.reduce_mean(self.loss)

        # # evaluation
        # logits = tf.matmul(inputs, tf.transpose(weights))
        #     logits = tf.nn.bias_add(logits, biases)
        #     labels_one_hot = tf.one_hot(labels, n_classes)
        #     loss = tf.nn.softmax_cross_entropy_with_logits(
        #         labels=labels_one_hot,
        #         logits=logits)

        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        # testing
        self.test_probs = tf.nn.softmax(self.logits) # shape: [batch_size x unique_jobs]

        return self

    def save_model(self, session, model_name):
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        saver = tf.train.Saver()
        saver.save(session, 'saved_models/' + model_name + '/' + 'model.checkpoint')

    def nemo_mpr(self,y_pred_proba,y_true,class_labels):
        mpr = np.mean([np.where(class_labels[y_pred_proba[i].argsort()[::-1]] == y_true[i])[0][0] / len(class_labels)
                       for i in range(len(y_true))
                       if y_true[i] in class_labels])
        return mpr

    def train_nemo_model(self, n_iter, print_freq, model_name):
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
                train_loss = self.sess.run(self.loss,train_feed_dict)
                print('Train Loss at', iter, ": ", train_loss)

        # saving model
        self.save_model(self.sess,model_name)

        return self


    # TODO: test this
    def restore_nemo_model(self, model_name):
        tf.reset_default_graph()
        save_dir = 'saved_models/' + model_name + '/'
        saver = tf.train.import_meta_graph(save_dir + 'model.checkpoint.meta')
        ckpt = tf.train.get_checkpoint_state(save_dir)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        # https://stackoverflow.com/questions/42832083/tensorflow-saving-restoring-session-checkpoint-metagraph

        return self

    def evaluate_nemo(self):
        # evaluate relevant variables from compute graph
        print('evaluating...')
        test_feed_dict = {self.max_pool_skills: self.X_skill_test,
                          self.job_inputs: self.X_job_test,
                          self.job_true: np.expand_dims(self.y_test,axis=1)}
        test_loss,test_probs = self.sess.run([self.loss,self.test_probs],test_feed_dict)
        print('Test Loss: ', test_loss)

        # calculating MPR
        print('calculating MPR')
        class_labels = np.array(range(self.n_unique_jobs))
        mpr = self.nemo_mpr(test_probs,self.y_test,class_labels)

        return mpr

    # TODO: complete this
    def test_individual_examples(self):
        # take random example from initial df

        # convert this into terms that NEMO would understand

        # run through the compute graph

        # output a prob

        # use reverse dict to convert into prediction
        pass


if __name__ == "__main__":

    model = NEMO(n_files=2)
    # model.restore_nemo_model(model_name='first_run')
    model.train_nemo_model(n_iter=10000,print_freq=1000,model_name='second_run')
    mpr = model.evaluate_nemo()
    print('MPR:',mpr)

    # # test mpr
    # test_array = np.random.rand(10,100)
    # y_true = np.random.randint(0,100,size=(10,))
    # class_labels = np.array(range(100))
    #
    # mpr = model.nemo_mpr(test_array,y_true,class_labels)
    # print('MPR is: ', mpr)