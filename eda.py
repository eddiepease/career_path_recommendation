import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from wordcloud import WordCloud

from read_data import read_json_data,read_ontology_data
from job_title_normalizer.ad_parsing import JobTitleNormalizer

class graph_object():

    def __init__(self, df):
        self.df = df
        self.years_bound = 60

    # transform methods
    def work_experience_months(self):
        df = self.df['total_months_work_exp'].astype('float').dropna()
        self.transformed_df = df[df < self.years_bound * 12]

    def work_experience_years(self):
        df = self.df['total_months_work_exp'].astype('float').dropna()
        self.transformed_df = df[df < self.years_bound * 12].floordiv(12.0).rename('total_years_work_exp')

    # TODO: fill out industry people currently work in
    def most_recent_industry(self):
        # extract most recent company from df
        most_recent_job_title = []
        for i in range(0, len(df)):
            if len(df['employment_history'][i]) > 0:
                try:
                    most_recent_job_title.append(df['employment_history'][i][0]['company_name'])
                except KeyError:
                    pass

        self.transformed_df = pd.DataFrame(most_recent_job_title, columns=['most_recent_company_name'])

        # company to industry mapping
        pass

    # TODO: fill out normalised job titles
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
        # TODO: complete cross reference against ontology
        print('reference against ontology...')

        # ontology = read_ontology_data('title-npt')
        # self.transformed_df = pd.DataFrame(most_recent_job_title, columns=['npt'])
        # self.transformed_df = self.transformed_df.join(ontology, on='npt', how='left', rsuffix='ont')

        # # temp saving
        # freq_df = pd.Series(most_recent_job_title).value_counts().to_dict()
        # pickle.dump(freq_df, open("test_job_freq.pkl","wb"))


    # TODO: rough location, depends on what people are doing
    def location(self):

        pass

    # plot methods
    def generate_histogram(self):
        ax = sns.distplot(self.transformed_df)
        return ax

    def generate_word_cloud(self):
        job_freq_dict = pickle.load(open("test_job_freq.pkl","rb"))
        wordcloud = WordCloud().generate_from_frequencies(job_freq_dict)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


if __name__ == '__main__':

    # read data
    df = read_json_data()

    # transform data
    graph = graph_object(df)
    graph.most_recent_job_title()

    print(len(df))
    print(graph.transformed_df.head(40))
    print(len(graph.transformed_df))

    # # plot graph
    # plt.show(ax)

    # word cloud test





