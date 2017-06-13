import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from wordcloud import WordCloud

from read_data import read_json_data,read_ontology_data,read_general_csv
from job_title_normalizer.ad_parsing import JobTitleNormalizer

class ExploratoryDataAnalysis():

    def __init__(self, df):
        self.df = df
        self.years_bound = 60

    ###############
    # transformation methods
    ###############

    def work_experience_months(self):
        df = self.df['total_months_work_exp'].astype('float').dropna()
        self.transformed_df = df[df < self.years_bound * 12]

    def work_experience_years(self):
        df = self.df['total_months_work_exp'].astype('float').dropna()
        self.transformed_df = df[df < self.years_bound * 12].floordiv(12.0).rename('total_years_work_exp')


    def most_recent_job_title(self):

        most_recent_job_title = []

        # set up the normalizer
        fnoun_plural = pickle.load(open("job_title_normalizer/data/fnoun_plural_dict.pkl", "rb"), encoding='latin1')
        fnoun_set = pickle.load(open("job_title_normalizer/data/fnoun_set.pkl", "rb"), encoding='latin1')
        spellchecker = pickle.load(open("job_title_normalizer/data/spellchecker_dict.pkl", "rb"), encoding='latin1')
        stopwords = pickle.load(open("job_title_normalizer/data/stopwords.pkl", "rb"), encoding='latin1')
        title = pickle.load(open("job_title_normalizer/data/title_dict.pkl", "rb"), encoding='latin1')
        token_sub = pickle.load(open("job_title_normalizer/data/token_sub_dict.pkl", "rb"), encoding='latin1')
        us_uk_spellchecker = pickle.load(open("job_title_normalizer/data/us_uk_spellchecker_dict.pkl", "rb"),
                                         encoding='latin1')

        job_title_normalizer = JobTitleNormalizer(stopwords, us_uk_spellchecker, spellchecker, fnoun_plural, title,
                                                  token_sub, fnoun_set)

        # loop through df
        for i in range(0, len(df)):
            if len(df['employment_history'][i]) > 0:
                try:
                    raw_title = df['employment_history'][i][0]['raw_job_title']
                    normalized_title = job_title_normalizer.process(raw_title)['title_norm']
                    most_recent_job_title.append(normalized_title)
                except KeyError:
                    pass

        # cross reference with ontology
        print('reference against ontology...')
        self.transformed_df = pd.DataFrame(most_recent_job_title, columns=['pt'])

        # temp saving
        freq_df = pd.Series(self.transformed_df['pt']).value_counts().to_dict()
        pickle.dump(freq_df, open("test_job_freq.pkl","wb"))


    # TODO: think about how to deal with the unknowns?
    def most_recent_job_category(self):

        # generate job title category
        ontology = read_ontology_data('title-category')
        category_df = read_general_csv('data/ontology/categories.csv')
        ontology.sort_values(['title', 'prob'], ascending=[True, False], inplace=True)
        max_category_ont = ontology.groupby('title').first()
        max_category_ont.reset_index(inplace=True)
        right_df = pd.merge(max_category_ont,category_df,how='inner',on='category_id')
        right_df.drop('prob', axis=1, inplace=True)

        # load in normalized job categories + join
        left_dict = pickle.load(open("test_job_freq.pkl", "rb"))
        left_df = pd.DataFrame(list(left_dict.items()),columns=['title','count'])
        merge_df = pd.merge(left_df, right_df, how='left', on='title')

        # final transformations
        self.transformed_df = merge_df.groupby(by='category_name')['count'].sum().reset_index()
        max_count = max(self.transformed_df['count'])
        self.transformed_df['count'] = self.transformed_df['count'] / max_count

    # TODO: rough location, depends on what people are doing
    def location(self):

        pass

    ##########
    # visualization methods
    ##########

    def generate_histogram(self):
        ax = sns.distplot(self.transformed_df)
        return ax

    def generate_word_cloud(self):
        job_freq_dict = pickle.load(open("test_job_freq.pkl","rb"))
        wordcloud = WordCloud().generate_from_frequencies(job_freq_dict)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def generate_bar_chart(self):
        ax = sns.barplot(x='count',y='category_name',data=self.transformed_df)
        return ax

    # TODO: adjust normalization for the Unknown
    def generate_industry_comparison_bar_chart(self):

        # preparing total industry
        total_industry_df = read_general_csv('data/manual/website_category_num.csv')
        max_count = max(total_industry_df['count'])
        total_industry_df['count'] = total_industry_df['count'] / max_count
        total_industry_df['type'] = 'website'

        # prepare transformed data
        self.transformed_df['type'] = 'cv'
        self.transformed_df = self.transformed_df[self.transformed_df['category_name'] != 'Unknown']
        self.transformed_df.sort_values('count',ascending=False,inplace=True)

        # join
        total_df = pd.concat([self.transformed_df,total_industry_df],axis=0)
        ax = sns.barplot(x='count', y='category_name', hue='type', data=total_df)

        return ax


if __name__ == '__main__':

    # read data
    df = read_json_data()

    # transform data
    graph = ExploratoryDataAnalysis(df)
    graph.most_recent_job_category()
    # print(graph.transformed_df.head(30))
    #
    ax = graph.generate_industry_comparison_bar_chart()

    plt.show(ax)


    # graph.most_recent_job_title()
    # graph.generate_word_cloud()
    # graph.most_recent_job_category()


    # print('Number of rows: ',len(df))
    # print('Number of NaNs', graph.transformed_df['pt'].isnull().sum())
    # print(graph.transformed_df.head(100))
    # print(len(graph.transformed_df))




    # # plot graph
    # plt.show(ax)

    # word cloud test





