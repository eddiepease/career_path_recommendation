import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_data import read_json_data
from eda import ExploratoryDataAnalysis

# TODO: test this
def create_train_test_set(n_seed=1, train_frac=0.8):

    np.random.seed(n_seed) # set random seed
    df = read_json_data(folder='data/cvs/') # read in df

    msk = np.random.rand(len(df)) < train_frac
    train = df[msk]
    test = df[~msk]

    return train, test

# TODO: complete this
def plot_comparison_graphs(ax_1,ax_2,title,save_location):
    fig, axarr = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=16)

    fig.axes.append(ax_1)
    fig.axes.append(ax_2)

    # axarr[0].append(ax_1)
    # axarr[0].set_title('Training set')
    # axarr[1] = ax_2
    # axarr[1].set_title('Test set')

    # # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # Tight layout often produces nice results
    # but requires the title to be spaced accordingly
    fig.tight_layout()
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
    ax_1 = eda_1.generate_histogram()
    ax_2 = eda_2.generate_histogram()
    plot_comparison_graphs(ax_1,ax_2,title='Years in employment',
                           save_location='figures/compare_datasets/years_in_employment_comp.png')

    # # number of roles
    # print('working out number of roles...')
    # eda_1.number_of_roles()
    # eda_2.number_of_roles()
    # ax_1 = eda_1.generate_histogram()
    # ax_2 = eda_2.generate_histogram()
    # plot_comparison_graphs(ax_1, ax_2, title='Number of roles')
    #
    # # most recent job title
    # print('working out most recent job title...')
    # eda_1.most_recent_job_title(file_name='train_job_dict')
    # eda_2.most_recent_job_title(file_name='test_job_dict')
    # # ax_1 = eda_1.generate_histogram()
    # # ax_2 = eda_2.generate_histogram()
    # # plot_comparison_graphs(ax_1, ax_2, title='Most frequent job titles')
    #
    # # most recent job category
    # print('working out most recent job category')
    # eda_1.most_recent_job_category(job_title_filename='train_job_dict')
    # eda_2.most_recent_job_category(job_title_filename='test_job_dict')
    # ax_1 = eda_1.generate_bar_chart()
    # ax_2 = eda_2.generate_bar_chart()
    # plot_comparison_graphs(ax_1,ax_2, title='Job Categories')
    #
    # # attended university
    # print('working out % that attended university...')
    # eda_1.attended_university()
    # eda_2.attended_university()
    # ax_1 = eda_1.generate_bar_chart()
    # ax_2 = eda_2.generate_bar_chart()
    # plot_comparison_graphs(ax_1, ax_2, title='University Attendance')
    #
    # # location - need to compare this manually
    # print('working out location...')
    # eda_1.location(file_location='figures/compare_datasets/train_map.html')
    # eda_2.location(file_location='figures/compare_datasets/train_map.html')



if __name__ == "__main__":

    train,test = create_train_test_set()
    compare_cv_dfs(train,test)


