import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score
)

class ResultStore:
    """A centralized dictionary-based store for all experiment metrics."""

    def __init__(self):
        self.results = []

    def add_result(self, dataset_name, embedding_name, algo_name, metrics):
        result_entry = {
            "Dataset": dataset_name,
            "Embedding": embedding_name,
            "Algorithm": algo_name,
            **metrics
        }
        self.results.append(result_entry)

    def get_summary(self):
        return pd.DataFrame(self.results)


class ClusterEvaluator:
    """Calculates and stores performance metrics for clustering."""

    @staticmethod
    def evaluate(data, true_labels, predicted_labels):
        # Handle HDBSCAN noise (-1) as a separate cluster for metric calculation
        # Most metrics handle this fine, but it's important to be aware of.

        metrics = {
            "NMI": normalized_mutual_info_score(true_labels, predicted_labels),
            "ARI": adjusted_rand_score(true_labels, predicted_labels),
            "AMI": adjusted_mutual_info_score(true_labels, predicted_labels),
        }

        # Silhouette score requires at least 2 clusters and doesn't work with 'noise'
        # points labeled as -1 by HDBSCAN if they are the only points in a 'cluster'.
        unique_labels = len(set(predicted_labels) - {-1})
        if unique_labels > 1:
            metrics["Silhouette"] = silhouette_score(data, predicted_labels)
        else:
            metrics["Silhouette"] = 0.0

        return metrics