import pandas as pd
import numpy as np
from pathlib import Path
import hdbscan

DATA_PATH = Path("data/processed/cleaned_news_dataset.csv")
EMBEDDING_PATH = Path("data/processed/news_embeddings.npy")
OUTPUT_PATH = Path("outputs/clusters/clustered_news.csv")

def load_files():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDING_PATH)

    # Keep only first 10,000 rows
    df = df.head(len(embeddings))

    print(f"Dataset Shape    : {df.shape}")
    print(f"Embeddings Shape : {embeddings.shape}")

    return df, embeddings

#HDBSCAN model
def build_cluster_model():

    model = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="leaf"
    )

    return model

# Perform Clustering
def perform_clustering(model, embeddings):
    print("\nCreating clusters...")
    cluster_labels = model.fit_predict(embeddings)
    print("Clustering Completed!")
    return cluster_labels

# Add Cluster IDs
def add_cluster_labels(df, cluster_labels):
    df["cluster_id"] = cluster_labels
    return df


# Cluster Statistics
def show_cluster_statistics(df):
    print("\nCluster Statistics")
    total_clusters = len(
        df[df["cluster_id"] != -1]["cluster_id"].unique()
    )
    noise_articles = len(df[df["cluster_id"] == -1])

    print(f"Total Clusters : {total_clusters}")
    print(f"Noise Articles : {noise_articles}")

    print("\nTop 10 Cluster Sizes\n")
    print(df["cluster_id"].value_counts().head(10))

# Save Output

def save_clusters(df):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nClustered dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    news_df, news_embeddings = load_files()
    cluster_model = build_cluster_model()
    labels = perform_clustering(
        cluster_model,
        news_embeddings
    )
    news_df = add_cluster_labels(
        news_df,
        labels
    )
    show_cluster_statistics(news_df)
    save_clusters(news_df)
    print("\nEvent Clustering Completed Successfully!")