import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from read_data import read_ontology_data,read_embeddings_json

# TODO: complete this
def convert_skills_profile_to_dict():
    pass

# TODO: alter this to take account of the function above when completed
def create_job_embedding(embedding_size):

    # read in skills profiles + order + normalize TD-IDF scores
    skills_profile_df = read_ontology_data('skill-profiles')
    skills_profile_df.sort_values(['title', 'weight'], ascending=[False, False], inplace=True)
    skills_profile_df.reset_index(drop=True, inplace=True)
    skills_profile_df = skills_profile_df.assign(
        normalized=skills_profile_df['weight'].div(skills_profile_df.groupby('title')['weight'].transform('sum')))

    # read in skills embedding
    file_name = 'data/ontology/skill-word2vec/data/skill_embeddings.json'
    skill_embeddings_dict = read_embeddings_json(file_name)

    # average skills to create job embedding
    unique_job_titles = list(skills_profile_df['title'].unique())
    data = np.empty(shape=(len(unique_job_titles), embedding_size), dtype=np.float32)

    # merge these together to create a numpy array
    for i, job in enumerate(unique_job_titles):
        print(i)
        job_array = np.zeros(shape=(1, embedding_size))
        job_df = skills_profile_df[skills_profile_df['title'] == job]
        skills = list(job_df['skill'])
        norm_weights = list(job_df['normalized'])

        for j, skill in enumerate(skills):
            job_array = job_array + skill_embeddings_dict[skill] * norm_weights[j]

        data[i, :] = job_array

    return data, unique_job_titles


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

def evaluate_with_tsne(embedding, num_to_plot, input_labels, filename):
    tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000)

    labels = input_labels[:num_to_plot]
    low_dim_embs = tsne.fit_transform(embedding[:num_to_plot, :])
    plot_with_labels(low_dim_embs, labels, filename)



if __name__ == "__main__":

    # TODO: something gimmicky with vector maths
    # TODO: think about a good way to evaluate the embeddings in a more rigorous way
    # TODO: is there a different way to generate the embeddings

    # create job embedding
    job_embedding, job_titles = create_job_embedding(embedding_size=100) # this must be the same as skill embedding

    #plot using TSNE
    evaluate_with_tsne(job_embedding,num_to_plot=800, input_labels=job_titles, filename='job_tsne.png')

