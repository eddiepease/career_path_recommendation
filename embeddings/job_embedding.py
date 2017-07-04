import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from read_data import read_ontology_data,read_embeddings_json

# TODO: test this
def create_job_embedding(embedding_size):

    # read in skills profiles + order + normalize TD-IDF scores
    skill_profile_dict = read_ontology_data('skill-profiles')

    # read in skills embedding
    file_name = 'data/ontology/skill-word2vec/data/skill_embeddings.json'
    skill_embeddings_dict = read_embeddings_json(file_name)

    # initialize numpy array
    data = np.empty(shape=(len(skill_profile_dict), embedding_size), dtype=np.float32)

    # merge these together to create a numpy array
    for i,value in enumerate(skill_profile_dict.values()):
        print(i)
        job_array = np.zeros(shape=(1, embedding_size))
        skills = value[0]
        norm_weights = value[2]

        for j, skill in enumerate(skills):
            job_array = job_array + skill_embeddings_dict[skill] * norm_weights[j]

        data[i, :] = job_array

    return data, skill_profile_dict.keys()


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
    job_embedding, job_titles = create_job_embedding(embedding_size=100) # this must be the same size as skill embedding

    #plot using TSNE
    evaluate_with_tsne(job_embedding,num_to_plot=800, input_labels=job_titles, filename='job_tsne.png')

