import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_PATH = Path("outputs/clusters/clustered_news.csv")
OUTPUT_PATH = Path("outputs/clusters/labeled_news.csv")


def load_dataset():
    print("Loading clustered dataset...")
    df = pd.read_csv(INPUT_PATH)
    return df

def generate_label(cluster_articles):
    vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    tfidf_matrix = vectorizer.fit_transform(cluster_articles)
    keywords = vectorizer.get_feature_names_out()
    label = " | ".join(keywords[:3])
    return label


def create_event_labels(df):
    event_labels = {}
    clusters = sorted(df["cluster_id"].unique())
    for cluster in clusters:
        if cluster == -1:
            event_labels[cluster] = "Noise / Miscellaneous"
            continue

        articles = df[df["cluster_id"] == cluster]["text"]
        label = generate_label(articles)
        event_labels[cluster] = label

    df["event_label"] = df["cluster_id"].map(event_labels)
    return df, event_labels


def show_labels(event_labels):
    print("\nGenerated Event Labels\n")
    for cluster, label in event_labels.items():
        print(f"Cluster {cluster} : {label}")


def save_dataset(df):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved labeled dataset to {OUTPUT_PATH}")
    EVENT_NAME_MAP = {
    "coronavirus | covid | face": "COVID-19 Pandemic",
    "jong | kim | korea": "North Korea – Kim Jong-un",
    "death | earthquake | indonesia": "Indonesia Earthquake",
    "best | movies | netflix": "Netflix Movies & Entertainment",
    "abortion | abortions | ban": "US Abortion Ban Debate",
    "biden | democrats | joe": "Joe Biden & US Politics"}
    df["event_label"] = df["event_label"].replace(EVENT_NAME_MAP)

if __name__ == "__main__":
    news_df = load_dataset()
    news_df, labels = create_event_labels(news_df)
    show_labels(labels)
    save_dataset(news_df)
    print("\nEvent Labeling Completed Successfully!")