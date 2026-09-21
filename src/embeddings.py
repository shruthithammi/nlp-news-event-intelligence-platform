import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np


INPUT_PATH = Path("data/processed/cleaned_news_dataset.csv")
OUTPUT_PATH = Path("data/processed/news_embeddings.npy")


def load_dataset():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH)
    return df


def load_model():
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model Loaded Successfully!")
    return model

def generate_embeddings(model, texts):
    print("\nGenerating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print("Embeddings Generated!")
    print("Embedding Shape :", embeddings.shape)
    return embeddings

def save_embeddings(embeddings):
    np.save(OUTPUT_PATH, embeddings)
    print(f"\nSaved embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    news_df = load_dataset()
    model = load_model()
    embeddings = generate_embeddings(
        model,
        news_df["text"].head(10000).tolist()
    )
    save_embeddings(embeddings)
    print("\nEmbedding Module Completed Successfully!")