import os
import pandas as pd
from src import (
    DatasetManager, EmbeddingEngine, Clusterer,
    ClusterEvaluator, ResultStore, Visualizer
)


def run_experiment():
    # 1. Initialization

    # Create output directories if they don't exist
    os.makedirs("outputs/plots", exist_ok=True)
    bbc_path = 'Datasets/bbc_news_test.csv'
    db_manager = DatasetManager(bbc_path)
    embed_engine = EmbeddingEngine(vector_size=100)
    clusterer = Clusterer()
    evaluator = ClusterEvaluator()
    results = ResultStore()
    viz = Visualizer()

    # Load both datasets
    datasets = db_manager.prepare_data()

    # Define our search space
    embedding_methods = ['tfidf', 'word2vec', 'fasttext']
    clustering_methods = ['kmeans', 'agglomerative', 'hdbscan']

    # 2. The Experiment Loop
    for ds_name, df in datasets.items():
        print(f"\n{'=' * 30}\nProcessing Dataset: {ds_name.upper()}\n{'=' * 30}")

        # Ground truth labels and number of classes
        true_labels = df['label'] if 'label' in df.columns else pd.factorize(df['category'])[0]
        n_classes = len(set(true_labels))

        for embed_type in embedding_methods:
            # Generate Embeddings
            if embed_type == 'tfidf':
                X = embed_engine.get_tfidf_embeddings(df['cleaned_text'])
            elif embed_type == 'word2vec':
                X = embed_engine.get_word2vec_embeddings(df['cleaned_text'])
            else:
                X = embed_engine.get_fasttext_embeddings(df['cleaned_text'])

            for algo in clustering_methods:
                # Run Clustering
                # Note: HDBSCAN doesn't need n_clusters, but K-Means and Agglomerative do
                predicted = clusterer.run(algo, X, n_clusters=n_classes)

                # Evaluate
                metrics = evaluator.evaluate(X, true_labels, predicted)
                results.add_result(ds_name, embed_type, algo, metrics)

                print(f"Result: {embed_type} + {algo} -> NMI: {metrics['NMI']:.3f}")

            # 3. Visualization Phase (Post-Evaluation)
            # Reduce and Visualize the last algorithm's result for this embedding
            coords_2d = viz.reduce_dimensions(X)
            viz.plot_clusters(
                coords_2d, predicted,
                title=f"{ds_name.upper()}: {embed_type} + {algo}",
                save_path=f"outputs/plots/{ds_name}_{embed_type}_{algo}.png"
            )

    # 4. Final Reporting
    final_df = results.get_summary()
    final_df.to_csv("outputs/results.csv", index=False)
    print("\nFinal Performance Summary:")
    print(final_df)

    # Plot all metrics side-by-side
    viz.plot_comparison_metrics(final_df, metrics=["NMI", "ARI", "AMI", "Silhouette"])

    # # Generate overall comparison plot
    # viz.plot_comparison(final_df, metric="NMI")

if __name__ == "__main__":
    run_experiment()