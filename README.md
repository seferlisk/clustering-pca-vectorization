# NLP Clustering Benchmark: BBC News vs. 20NewsGroups

An object-oriented machine learning pipeline designed to compare frequency-based (TF-IDF) and prediction-based (Word2Vec, FastText) embeddings using various clustering techniques.

##  Project Overview
This project explores how different text vectorization methods impact the quality of unsupervised clustering. We evaluate performance across two distinct datasets using a combination of centroid-based, density-based, and hierarchical algorithms.

### Key Features
* **Multi-Model Support:** TF-IDF, Word2Vec, and FastText.
* **Clustering Suite:** K-Means, HDBSCAN, and Agglomerative Clustering.
* **Dimensionality Reduction:** Integrated PCA for noise reduction and t-SNE for 2D visualization.
* **OOP Architecture:** Modular design for easy experimentation and scalability.

---

## 📁 Project Structure
```text
ml_clustering_project/
├── src/
│   ├── preprocessor.py    # Text cleaning & tokenization
│   ├── data_loader.py     # Dataset management (BBC & 20News)
│   ├── embeddings.py      # Embedding generation (TF-IDF, W2V, FastText)
│   ├── clustering.py      # Clustering algorithm implementations
│   ├── evaluation.py      # Performance metrics & result storage
│   └── visualizer.py      # PCA, t-SNE, and Plotting
├── data/                  # Local datasets (e.g., bbc-text.csv)
├── outputs/               # Generated plots and results.csv
└── main.py                # Pipeline entry point