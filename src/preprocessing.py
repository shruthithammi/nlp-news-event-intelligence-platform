import pandas as pd
import re
from pathlib import Path
INPUT_PATH = Path("data/processed/news_dataset.csv")
OUTPUT_PATH = Path("data/processed/cleaned_news_dataset.csv")

def load_dataset():
    print("Loading processed dataset...")
    df = pd.read_csv(INPUT_PATH)
    print("Dataset Loaded.")
    return df

#remove missing values
def remove_missing_values(df):
    print("\nRemoving missing values...")
    df = df.dropna(subset=["title", "description", "date"])
    print(f"Remaining rows : {len(df)}")
    return df


def remove_duplicates(df):
    print("\nRemoving duplicate articles...")
    df = df.drop_duplicates(subset=["title"])
    print(f"Remaining rows : {len(df)}")
    return df

def clean_text(text):
    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def preprocess_text(df):
    print("\nCleaning title and description...")
    df["title"] = df["title"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)
    return df



def convert_date(df):
    print("\nConverting date format...")
    df["date"] = pd.to_datetime(df["date"])
    return df



def create_text_column(df):
    print("\nCreating text column...")
    df["text"] = df["title"] + " " + df["description"]
    return df

def save_dataset(df):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    news_df = load_dataset()
    news_df = remove_missing_values(news_df)
    news_df = remove_duplicates(news_df)
    news_df = preprocess_text(news_df)
    news_df = convert_date(news_df)
    news_df = create_text_column(news_df)
    save_dataset(news_df)
    print("\nPreprocessing Completed Successfully!")