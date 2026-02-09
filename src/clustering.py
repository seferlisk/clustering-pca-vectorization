from sklearn.cluster import KMeans, AgglomerativeClustering
import hdbscan
import numpy as np


class Clusterer:
    """
    A unified interface for different clustering algorithms.
    """

    def __init__(self):
        self.models = {
            'kmeans': self._run_kmeans,
            'agglomerative': self._run_agglomerative,
            'hdbscan': self._run_hdbscan
        }

    def run(self, algorithm_name, data, n_clusters=None):
        """Dispatches the data to the correct algorithm."""
        if algorithm_name not in self.models:
            raise ValueError(f"Algorithm {algorithm_name} not supported.")

        return self.models[algorithm_name](data, n_clusters)

    def _run_kmeans(self, data, n_clusters):
        print(f"Running K-Means with k={n_clusters}...")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return model.fit_predict(data)

    def _run_agglomerative(self, data, n_clusters):
        print(f"Running Agglomerative Clustering with k={n_clusters}...")
        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(data)

    def _run_hdbscan(self, data, n_clusters=None):
        print("Running HDBSCAN...")
        # n_clusters is ignored for HDBSCAN, it uses min_cluster_size
        model = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
        return model.fit_predict(data)