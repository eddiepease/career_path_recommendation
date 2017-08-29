import time
import numpy as np
from scipy.spatial.distance import hamming
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from baseline_model import BaselineModel
from embeddings.job_embedding import create_job_embedding
from split_data import create_train_test_set_stratified_baseline


def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if (not hasattr(estimator, "decision_function") and
            not hasattr(estimator, "predict_proba")):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


# TODO: sort out definition of the classifer
# class which implements ECOC method
class ECOC(BaselineModel):

    # TODO: add multithreading with different cores
    def __init__(self, train, test, estimator, n_classifiers=10):
        BaselineModel.__init__(self,train,test)
        self.estimator = estimator
        self.n_classifiers = n_classifiers

    # method to create the code book which the classifier uses
    # TODO: problem to investigate - number of duplicate rows
    def create_code_book(self,save_name):
        print('creating code book...')

        # load data into object
        self.load_transformed_data(save_name)

        # import job embedding
        self.embedding, self.ordered_job_title = create_job_embedding(embedding_size=100)

        # need to write a line which selects the relevant parts of the embedding
        idx = np.sort(np.unique(self.y_train))
        self.embedding = self.embedding[idx]
        self.ordered_job_title = [self.ordered_job_title[i] for i in idx]

        # set up a blank array
        code_book = np.ones(shape=(len(self.ordered_job_title),self.n_classifiers))

        # loop through columns in cookbook
        for i in range(self.n_classifiers):
            col_imbalance = code_book[:,i].sum() / code_book.shape[0]
            error_margin = 0.05
            while abs(col_imbalance) > error_margin:
                w = np.random.rand(100,1) * 2 - 1
                code_col = np.squeeze(np.dot(self.embedding,w),axis=1)
                code_col = np.sign(code_col)

                col_imbalance = code_col.sum() / code_book.shape[0]
                code_book[:,i] = code_col

        self.code_book_ = code_book

        # # code to output same codes
        # test_code = np.array([-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1., -1.,  1.,  1., -1., 1., -1.])
        # for i,row in enumerate(code_book):
        #     if row == test_code.all():
        #         job_title = self.ordered_job_title[i]
        #         print(job_title)

        return self

    # method to fit the classifier
    # TODO: run binary classifiers on many CPUs
    def fit(self):
        print('fitting ecoc classifiers...')

        _check_estimator(self.estimator)

        # set up
        self.classes_ = np.unique(self.y_train)
        # n_classes = self.classes_.shape[0]
        self.classes_index = dict((c, j) for j, c in enumerate(self.classes_))
        Y = np.array([self.code_book_[self.classes_index[self.y_train[i]]]
                      for i in range(self.X_train.shape[0])], dtype=np.int)

        # fit the binary classifiers
        t0 = time.time()
        # estimators = []
        # for k in range(self.n_classifiers):
        #     curr_est = self.estimator.fit(self.X_train, Y[:,k])
        #     estimators.append(curr_est)
        self.estimators_ = [SVC().fit(self.X_train, Y[:, k]) for k in range(0, self.n_classifiers)]
        t1 = time.time()
        print('Training time for classifiers:', t1-t0)

        return self

    # TODO: work out why results isn't as good as expected
    # TODO: different evaluation possible?
    def predict_mpr(self):
        print('Predicting mpr...')

        # initialise mpr list
        pr_list = []

        classifier_output = np.array([e.predict(self.X_test) for e in self.estimators_]).T

        # loop through all the rows in test
        for i,row in enumerate(classifier_output):

            # calculate MPR score
            hamm_dist = np.array([hamming(row, self.code_book_[j, :]) for j in range(0, len(self.code_book_))])
            temp = hamm_dist.argsort()
            ranks = np.empty(len(hamm_dist), int)
            ranks[temp] = np.arange(start=1, stop=len(hamm_dist) + 1)
            pr = ranks[self.classes_index[self.y_test[i]]] / len(hamm_dist)
            pr_list.append(pr)

        mpr = np.mean(pr_list)

        return mpr


if __name__ == "__main__":

    # create train/test set
    train, test = create_train_test_set_stratified_baseline(n_files=1, threshold=1)

    # test ECOC
    folder = 'ecoc_test'
    ecoc = ECOC(train,test,estimator=LogisticRegression(),n_classifiers=15)
    # ecoc.save_transformed_data(embedding=True, weighted=False, save_name=folder)
    ecoc.create_code_book(save_name=folder)
    ecoc.fit()
    mpr = ecoc.predict_mpr()
    print('MPR is:', mpr)

