import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from read_data import read_embeddings_json

# plot TSNE
def plot_with_labels(low_dim_embs, labels, filename='figures/skill_tsne.png'):
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

def evaluate_with_tsne(embedding_dict, num_to_plot):
    tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000)

    embedding_df = pd.DataFrame(embedding_dict)
    # print(embedding_df.head(10))
    labels = embedding_df.columns[:num_to_plot]
    final_embedding = np.transpose(np.array(embedding_df))
    # print(final_embedding[:10,:])

    low_dim_embs = tsne.fit_transform(final_embedding[:num_to_plot, :])
    # labels = [self.reversed_job_dict[i] for i in range(len(plot_labels))]
    plot_with_labels(low_dim_embs, labels)

if __name__ == "__main__":

    # read in file
    file_name = 'data/ontology/skill-word2vec/data/skill_embeddings.json'
    skill_embedding = read_embeddings_json(file_name)

    # evaluate tsne
    evaluate_with_tsne(skill_embedding, num_to_plot=1000)
