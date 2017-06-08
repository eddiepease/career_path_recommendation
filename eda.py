import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from read_data import read_json_data

class transformed_df():

    def __init__(self, full_df):
        self.full_df = full_df
        self.experience_percentile = 99.5

    def work_experience_months(self):
        df = self.full_df['total_months_work_exp'].astype('float').dropna()
        transformed_df = df[df < np.percentile(df, self.experience_percentile)]
        return transformed_df

    def work_experience_years(self):
        df = self.full_df['total_months_work_exp'].astype('float').dropna()
        transformed_df = df[df < np.percentile(df, self.experience_percentile)].floordiv(12.0).rename('total_years_work_exp')
        return transformed_df


class created_graph():

    def __init__(self,transformed_df):
        self.transformed_df = transformed_df
        self.name = self.transformed_df.name

    def generate_histogram(self):
        ax = sns.distplot(self.transformed_df)
        return ax


if __name__ == '__main__':

    # read data
    df_full = read_json_data()

    # transform data
    df = transformed_df(df_full)
    df_trans = df.work_experience_years()

    # create graph
    graph = created_graph(df_trans)
    ax = graph.generate_histogram()

    # plot graph
    plt.show(ax)




