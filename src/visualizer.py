import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Visualizer:
    """
    Handles dimensionality reduction and plot generation.
    """

    def __init__(self):
        sns.set_theme(style="whitegrid", palette="viridis")

    def reduce_dimensions(self, data, n_pca=50):
        """
        Reduces data to 2D using PCA followed by t-SNE for
        optimal interpretability.
        """
        print(f"Reducing dimensions: PCA({n_pca}) -> t-SNE(2)...")

        # PCA for noise reduction
        n_comp = min(n_pca, data.shape[1], data.shape[0])
        pca = PCA(n_components=n_comp)
        pca_result = pca.fit_transform(data)

        # t-SNE for 2D mapping
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
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
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_comparison_metrics(self, results_df, metrics=None):
        """Creates faceted bar plots comparing various metrics
        across algorithms, embeddings, and datasets."""

        if metrics is None:
            metrics = ["NMI", "ARI", "AMI", "Silhouette"]

        for metric in metrics:
            print(f"Generating comparison plot for {metric}...")
            g = sns.catplot(
                data=results_df, kind="bar",
                x="Algorithm", y=metric, hue="Embedding",
                col="Dataset", palette="viridis", alpha=.8, height=5
            )
            g.despine(left=True)
            g.set_axis_labels("Clustering Algorithm", metric)
            g.legend.set_title("Embedding Type")

            # Save each metric plot separately
            plt.savefig(f"outputs/plots/comparison_{metric.lower()}.png")
            plt.close()