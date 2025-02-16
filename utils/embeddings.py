import numpy as np
import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def plot(df_emb: pandas.DataFrame, n_clusters: int):
    matrix = np.vstack(df_emb.embedding.values)
    matrix.shape
    # https://cookbook.openai.com/examples/clustering

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df_emb["Cluster"] = labels
    # df_emb.columns


    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    for category, color in tqdm(enumerate(["purple", "green", "red", "blue"])):
        xs = np.array(x)[df_emb.Cluster == category]
        ys = np.array(y)[df_emb.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified visualized in language 2d using t-SNE")
