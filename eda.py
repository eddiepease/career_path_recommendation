import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from wordcloud import WordCloud
from fuzzywuzzy import fuzz
import gmplot

from read_data import read_all_json_data,read_ontology_data,read_general_csv, CVJobNormalizer,read_single_json_data


# class used for any exploratory data analysis - both transformation and visualization methods
class ExploratoryDataAnalysis():

    def __init__(self, df, job_title_location):
        self.df = df
        self.years_bound = 60
        self.job_title_location = job_title_location

    ###############
    # transformation methods
    ###############

    def work_experience_months(self):
        temp_df = self.df['total_months_work_exp'].astype('float').dropna()
        self.transformed_df = temp_df[temp_df < self.years_bound * 12]

    def work_experience_years(self):
        temp_df = self.df['total_months_work_exp'].astype('float').dropna()
        self.transformed_df = temp_df[temp_df < self.years_bound * 12].floordiv(12.0).rename('total_years_work_exp')

    def number_of_roles(self):
        temp_df = self.df['employment_history_norm'].apply(len)
        temp_df = pd.DataFrame(temp_df.value_counts()).reset_index()
        temp_df = temp_df[temp_df['index'] <= 20]
        temp_df.sort_values('index',inplace=True)
        temp_df['employment_history_norm'] = temp_df['employment_history_norm'] / temp_df['employment_history_norm'].max()
        self.transformed_df = temp_df


    def most_recent_job_title(self, file_name, job_num=0,normalizer_required=False):
        print('Start most_recent_job_title method...')

        # job_num defaults to most recent job (=0), 2nd most recent job - job_num=1 etc..

        most_recent_job_title = []

        if normalizer_required:
            cv_job_normalizer = CVJobNormalizer()

            # loop through df
            for i in range(0, len(self.df)):
                most_recent_job_title.append(cv_job_normalizer.normalized_job(df=self.df, n_row=i,job_num=job_num))

        else:
            for j in range(0,len(self.df)):
                print(j)
                if isinstance(df['employment_history_norm'][j], list):
                    if len(df['employment_history_norm'][j]) > 0:
                        try:
                            norm_title = df['employment_history_norm'][j][job_num]['title_norm']
                            most_recent_job_title.append(norm_title)
                        except KeyError:
                            pass
                        except IndexError:
                            pass

        # cross reference with ontology
        print('reference against ontology...')
        self.transformed_df = pd.DataFrame(most_recent_job_title, columns=['pt'])

        # temp saving
        freq_df = pd.Series(self.transformed_df['pt']).value_counts().to_dict()
        pickle.dump(freq_df, open(self.job_title_location + file_name + ".pkl","wb"))

    def number_missing_job_titles(self):
        num_missing_titles = []
        for i in range(0, len(self.df)):
            print(i)
            missing_roles_counter = 0
            person_emp_list = df['employment_history_norm'][i]
            if isinstance(person_emp_list, list):
                if len(person_emp_list) > 0:
                    # loop through all roles
                    for j in range(0,len(person_emp_list)):
                        if 'title_norm' not in person_emp_list[j]:
                            missing_roles_counter += 1

            # append final counter to list
            num_missing_titles.append(missing_roles_counter)

        # convert to df
        self.transformed_df = pd.DataFrame(num_missing_titles, columns=['pt'])


    def most_recent_job_category(self, job_title_filename):
        print('Start most_recent_job_category method...')

        # generate job title category
        ontology = read_ontology_data('title-category')
        category_df = read_general_csv('data/ontology/categories.csv')
        ontology.sort_values(['title', 'prob'], ascending=[True, False], inplace=True)
        max_category_ont = ontology.groupby('title').first()
        max_category_ont.reset_index(inplace=True)
        right_df = pd.merge(max_category_ont,category_df,how='inner',on='category_id')
        right_df.drop('prob', axis=1, inplace=True)

        # load in normalized job categories + join
        left_dict = pickle.load(open(self.job_title_location + job_title_filename + ".pkl", "rb"))
        left_df = pd.DataFrame(list(left_dict.items()),columns=['title','count'])
        merge_df = pd.merge(left_df, right_df, how='left', on='title')

        # final transformations
        self.transformed_df = merge_df.groupby(by='category_name')['count'].sum().reset_index()
        self.transformed_df = self.transformed_df[self.transformed_df['category_name'] != 'Unknown']
        max_count = max(self.transformed_df['count'])
        self.transformed_df['count'] = self.transformed_df['count'] / max_count
        self.transformed_df.sort_values('count', ascending=False, inplace=True)
        self.transformed_df = self.transformed_df[['count','category_name']]

    def attended_university(self):
        print('Start attended_university method..')
        uni = []
        abbrev = ['ucl', 'lse', 'soas', 'uea', 'uwe']
        qual_list = ['master','bachelor','ba','ma','msc','bsc']
        for row in range(0, len(self.df)):
            num_entries = len(self.df['education_history'][row])
            uni_attend = False
            if num_entries > 0:
                for entry in range(0, num_entries):
                    try:
                        name = self.df['education_history'][row][entry]['institution_name'].lower()
                        qual = self.df['education_history'][row][entry]['qualification_type'].lower()
                        if fuzz.partial_ratio('university', name) > 80 or name in abbrev or \
                                any(substring in qual for substring in qual_list):
                            uni_attend = True
                            uni.append('University')
                            break

                    except KeyError:
                        pass

            if uni_attend == False:
                uni.append('No University')

        freq_df = pd.DataFrame(pd.Series(uni).value_counts()).reset_index()
        freq_df.columns = ['status','count']
        # print(freq_df)
        self.transformed_df = freq_df


    # note that this method saves a html file in figures folder
    # you need to screenshot locally to get a png
    def location(self,file_location):
        print('Start location method..')

        assert ('.html' in file_location), "Must save the map as html file!"

        # load postcodes ontology + merge with postcodes in the CVs
        post_ontology = read_general_csv('data/ontology/ukpostcodes.csv')
        left_df = self.df['postal_code'].dropna()
        left_df = pd.DataFrame(left_df.map(lambda x: str(x).replace(' ', '')))
        postcode_df = left_df.merge(post_ontology, how='left', left_on='postal_code', right_on='postcode')
        postcode_df = postcode_df[['postal_code', 'latitude', 'longitude']]
        postcode_df = postcode_df.dropna()

        # prepare for plot
        gmap = gmplot.GoogleMapPlotter.from_geocode('United Kingdom')

        lats = list(postcode_df['latitude'])
        lngs = list(postcode_df['longitude'])

        # create + save a heatmap
        gmap.heatmap(lats, lngs)
        gmap.draw(file_location)


    ##########
    # visualization methods
    ##########

    def generate_histogram(self,xlabel_name,axis=None):
        sns.distplot(self.transformed_df,ax=axis).set(xlabel=xlabel_name)

    def generate_word_cloud(self,file,title,save_location):
        job_freq_dict = pickle.load(open(self.job_title_location + "cvs_v4_job_freq.pkl","rb"))
        wordcloud = WordCloud(width=800,height=400).generate_from_frequencies(job_freq_dict)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(save_location + file,dpi=100)

    def generate_bar_chart(self,xlabel_name, axis=None):
        [first_column_name, second_column_name] = self.transformed_df.columns
        sns.barplot(x=first_column_name,y=second_column_name,data=self.transformed_df,ax=axis)\
            .set(xlabel=xlabel_name)
        # plt.savefig('figures/whole_university_attended.png')

    def generate_industry_comparison_bar_chart(self):

        # preparing total industry
        total_industry_df = read_general_csv('data/manual/website_category_num.csv')
        max_count_1 = max(total_industry_df['count'])
        total_industry_df['count'] = total_industry_df['count'] / max_count_1
        total_industry_df['type'] = 'website'

        # prepare transformed data
        self.transformed_df['type'] = 'cv'

        # join
        total_df = pd.concat([self.transformed_df,total_industry_df],axis=0)
        ax = sns.barplot(x='count', y='category_name', hue='type', data=total_df)

        return ax


if __name__ == '__main__':

    # read data
    df = read_all_json_data(folder='data/cvs_v4/')
    # df = read_single_json_data(num_file=1,folder='data/cvs_v4/')
    # print(df.columns)

    # transform data
    eda = ExploratoryDataAnalysis(df,job_title_location='data/')
    eda.number_of_roles()
    eda.generate_bar_chart(xlabel_name='Number of roles')

    plt.ylabel('Frequency')
    plt.title('Bar Chart of Number of Roles')
    plt.savefig('figures/whole_jobsite_data/cvs_v4/final_num_roles.png')

    # eda.most_recent_job_title(file_name='cvs_v4_job_freq')

    # eda.generate_word_cloud(file='final_job_wordcloud.jpg',
    #                         title='Job Title Wordcloud',
    #                         save_location='figures/whole_jobsite_data/cvs_v4/')
    # eda.most_recent_job_title(file_name='cvs_v3_job_freq')
    # eda.most_recent_job_category(job_title_filename='cvs_v3_job_freq')
    # eda.number_of_roles()
    # print(eda.transformed_df.shape)
    # print(eda.transformed_df.value_counts())
    # eda.work_experience_years()

    # # map plot
    # save_location = 'figures/whole_jobsite_data/cvs_v3/'
    # eda.location(file_location=save_location + 'cv_map.html')

    # # plot data
    # save_location = 'figures/whole_jobsite_data/cvs_v3/'
    # # eda.generate_word_cloud(file='job_wordcloud_test.png',title='Job Title Wordcloud',save_location=save_location)
    # eda.generate_bar_chart(xlabel_name='Normalized count')
    # plt.ylabel('Industry')
    # plt.title('Bar Chart of Job Categories')
    # plt.tight_layout()
    # plt.savefig(save_location + 'recent_job_category_test.png')

    # eda.generate_histogram(xlabel_name='Number of roles')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Number of Roles')
    # plt.savefig('number_of_roles.png')
    #
    # # # transform data
    # # eda = ExploratoryDataAnalysis(df,job_title_location='')
    # # eda.work_experience_years()
    # # eda.generate_histogram()
    # # plt.savefig('test.png')