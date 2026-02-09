import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


class Visualizer:
    """
    Handles dimensionality reduction and plot generation.
    """

    def __init__(self):
        sns.set_theme(style="whitegrid")

    def reduce_dimensions(self, data, n_pca=50):
        """
        Reduces data to 2D using PCA followed by t-SNE for
        optimal interpretability.
        """
        print(f"Reducing dimensions: PCA({n_pca}) -> t-SNE(2)...")

        # PCA handles the initial noise reduction
        # We handle sparse TF-IDF vs dense embeddings differently
        pca = PCA(n_components=min(n_pca, data.shape[1]))
        pca_result = pca.fit_transform(data)

        # t-SNE creates the 2D map
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        return tsne.fit_transform(pca_result)

    def plot_clusters(self, coords, labels, title, save_path):
        """Generates a 2D scatter plot of the clusters."""
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=coords[:, 0], y=coords[:, 1],
            hue=labels, palette='viridis',
            legend='full', alpha=0.6
        )
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    def plot_comparison(self, results_df, metric="NMI"):
        """Creates a bar plot comparing metrics across models."""
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df,
            x="Algorithm", y=metric,
            hue="Embedding"
        )
        plt.title(f"Performance Comparison: {metric}")
        plt.ylim(0, 1)
        plt.show()