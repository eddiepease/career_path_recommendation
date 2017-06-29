import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_data import read_json_data
from eda import ExploratoryDataAnalysis


def create_train_test_set(n_seed=1, train_frac=0.8):

    np.random.seed(n_seed) # set random seed
    df = read_json_data(folder='data/cvs/') # read in df

    msk = np.random.rand(len(df)) < train_frac
    train = df[msk].reset_index(drop=True)
    test = df[~msk].reset_index(drop=True)

    return train, test


def plot_comparison_graphs(object_1,object_2,xlabel_1,xlabel_2,title,save_location):
    fig, axs = plt.subplots(nrows=2,figsize=(10,10))
    fig.suptitle(title)  # , fontsize=16)
    object_1.current_plot_method(xlabel_name=xlabel_1,axis=axs[0])
    object_2.current_plot_method(xlabel_name=xlabel_2,axis=axs[1])
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_location)


# function to plot a series of graphs such that it is very easy to see the difference between 2 sets of CVs
def compare_cv_dfs(df_1,df_2):

    # setup
    print('setting up...')
    eda_1 = ExploratoryDataAnalysis(df=df_1,job_title_location='figures/compare_datasets/')
    eda_2 = ExploratoryDataAnalysis(df=df_2,job_title_location='figures/compare_datasets/')

    # years in employment
    print('working out years in employment...')
    eda_1.work_experience_years()
    eda_2.work_experience_years()
    eda_1.current_plot_method = eda_1.generate_histogram
    eda_2.current_plot_method = eda_2.generate_histogram
    plot_comparison_graphs(object_1=eda_1,
                           object_2=eda_2,
                           xlabel_1='number of years - train set',
                           xlabel_2='number of years - test set',
                           title='Histogram of Years of Work Experience',
                           save_location='figures/compare_datasets/work_experience_years.png')

    # TODO: complete this
    # # # number of roles
    # # print('working out number of roles...')
    # # eda_1.number_of_roles()
    # # eda_2.number_of_roles()
    # # ax_1 = eda_1.generate_histogram()
    # # ax_2 = eda_2.generate_histogram()
    # # plot_comparison_graphs(ax_1, ax_2, title='Number of roles')

    # most recent job title
    print('working out most recent job title...')
    file_1 = 'train_job_dict'
    file_2 = 'test_job_dict'
    eda_1.most_recent_job_title(file_name=file_1)
    eda_2.most_recent_job_title(file_name=file_2)
    eda_1.generate_word_cloud(file=eda_1.job_title_location + file_1 + '.pkl',
                              title='Train Wordcloud',
                              save_location='figures/compare_datasets/train_wordcloud.png')
    eda_2.generate_word_cloud(file=eda_2.job_title_location + file_2 + '.pkl',
                              title='Test Wordcloud',
                              save_location='figures/compare_datasets/test_wordcloud.png')

    # most recent job category
    print('working out most recent job category')
    eda_1.most_recent_job_category(job_title_filename='train_job_dict')
    eda_2.most_recent_job_category(job_title_filename='test_job_dict')
    eda_1.current_plot_method = eda_1.generate_bar_chart
    eda_2.current_plot_method = eda_2.generate_bar_chart
    plot_comparison_graphs(object_1=eda_1,
                           object_2=eda_2,
                           xlabel_1='relative freq of train set',
                           xlabel_2='relative freq of test set',
                           title='Bar chart of Job Categories',
                           save_location='figures/compare_datasets/job_category.png')

    # attended university
    print('working out % that attended university...')
    eda_1.attended_university()
    eda_2.attended_university()
    eda_1.current_plot_method = eda_1.generate_bar_chart
    eda_2.current_plot_method = eda_2.generate_bar_chart
    plot_comparison_graphs(object_1=eda_1,
                           object_2=eda_2,
                           xlabel_1='train set',
                           xlabel_2='test set',
                           title='Bar chart of University Attendance',
                           save_location='figures/compare_datasets/university_attendance.png')

    # location - need to compare this manually
    print('working out location...')
    eda_1.location(file_location='figures/compare_datasets/train_map.html')
    eda_2.location(file_location='figures/compare_datasets/test_map.html')



if __name__ == "__main__":

    train,test = create_train_test_set()
    compare_cv_dfs(train,test)


