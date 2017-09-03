import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from read_data import read_ontology_data,read_embeddings_json


def create_job_embedding(embedding_size):
    print('calculating job embeddings...')

    # read in skills profiles + order + normalize TD-IDF scores
    skill_profile_dict = read_ontology_data('skill-profiles',file_type='pkl')

    # read in skills embedding
    # file_name = 'data/ontology/skill-word2vec-json/part-00000-f545a814-9c2f-420f-a022-2dd3fc62c30b.json'
    file_name_old = 'data/ontology/skill-word2vec/data/skill_embeddings.json'
    skill_embeddings_dict = read_embeddings_json(file_name_old)

    # initialize numpy array
    data = np.empty(shape=(len(skill_profile_dict), embedding_size), dtype=np.float32)
    ordered_jobs = list(np.sort(list(skill_profile_dict.keys())))

    # merge these together to create a numpy array
    for i,key in enumerate(ordered_jobs):
        # print(i)
        job_array = np.zeros(shape=(1, embedding_size))
        skills = skill_profile_dict[key][0]
        norm_weights = skill_profile_dict[key][2]

        for j, skill in enumerate(skills):
            job_array = job_array + skill_embeddings_dict[skill] * norm_weights[j]

        data[i, :] = job_array

    return data, ordered_jobs

# function to save job embedding as dict
def save_job_embed_as_dict():
    data,ordered_jobs = create_job_embedding(embedding_size=100)

    # create dict
    job_embed_dict = {}

    for i,job in enumerate(ordered_jobs):
        job_embed_dict[job] = list(data[i,:])

    # save dict as json
    path = 'data/ontology/job-word2vec/'
    if not os.path.exists(path):
        os.makedirs(path)

    pickle.dump(job_embed_dict, open(path + "job_embedding.pkl", "wb"))



# function to create skill embedding
def create_cv_skill_embeddings(skills, skill_embeddings_dict):

    # initialize
    embedding_size = 100
    skill_embed = np.zeros(shape=(embedding_size, ), dtype=np.float32)

    # loop through the skills
    for skill in skills:
        try:
            skill_embed = skill_embed + skill_embeddings_dict[skill] * 1/len(skills)
        except KeyError:
            pass

    return skill_embed

# # function to create skill embedding
# def create_weighted_cv_skill_embeddings(skills, skill_embeddings_dict,norm_feat_job):
#
#     # initialize
#     embedding_size = 100
#     skill_embed = np.zeros(shape=(embedding_size, ), dtype=np.float32)
#
#     # read in skills profiles + order + normalize TD-IDF scores
#     skill_profile_dict = read_ontology_data('skill-profiles', file_type='pkl')
#
#     # loop through the skills
#     for i,skill in enumerate(skills):
#         feat_job_skills = skill_profile_dict[norm_feat_job]
#         if skill in feat_job_skills[0]:
#             skill_idx = feat_job_skills[0].index(skill)
#             try:
#                 skill_embed = skill_embed + skill_embeddings_dict[skill] * feat_job_skills[2][skill_idx]
#             except KeyError:
#                 print('Missing skill is: ', skill)
#                 pass
#
#     return skill_embed


def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def plot_with_colour(low_dim_embs, labels, index_1,filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    # plt.figure(figsize=(18, 18)) # in inches
    plt.figure()

    plt.scatter(low_dim_embs[:len(index_1),0],low_dim_embs[:len(index_1),1], color='b',label='Teaching Jobs')
    plt.scatter(low_dim_embs[len(index_1):, 0], low_dim_embs[len(index_1):, 1], color='g', label='Developer Jobs')

    plt.legend(loc='best')
    plt.title('t-SNE plot of Teaching and Developer Job Embeddings')

    plt.savefig(filename)
    # plt.show()


# def evaluate_with_tsne(embedding, num_to_plot, input_labels, filename):
#     tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000)
#
#     labels = input_labels[:num_to_plot]
#     low_dim_embs = tsne.fit_transform(embedding[:num_to_plot, :])
#     plot_with_labels(low_dim_embs, labels, filename)

def evaluate_with_tsne(embedding,job_titles,filename):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    # plot_only = 300
    teach_indices = [i for i, s in enumerate(job_titles) if 'teach' in s]
    develop_indices = [i for i, s in enumerate(job_titles) if 'developer' in s]
    plot_indices = teach_indices + develop_indices
    print('Number of teach/develop jobs:',len(plot_indices))

    teach_names = [s for i, s in enumerate(job_titles) if 'teach' in s]
    develop_names = [s for i, s in enumerate(job_titles) if 'developer' in s]
    plot_labels = teach_names + develop_names

    low_dim_embs = tsne.fit_transform(embedding[plot_indices, :])
    # labels = [self.reversed_job_dict[i] for i in range(len(plot_labels))]
    # plot_with_labels(low_dim_embs, plot_labels)
    plot_with_colour(low_dim_embs,plot_labels,teach_names,filename)



if __name__ == "__main__":

    # create job embedding
    job_embedding, job_titles = create_job_embedding(embedding_size=100) # this must be the same size as skill embedding

    #plot using TSNE
    # evaluate_with_tsne(job_embedding,num_to_plot=800, input_labels=job_titles, filename='job_tsne_2.png')
    evaluate_with_tsne(job_embedding,job_titles,filename='test_tsne.png')

    # save_job_embed_as_dict()

