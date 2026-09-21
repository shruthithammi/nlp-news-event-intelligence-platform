import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/News_Category_Dataset_v3.json")
OUTPUT_PATH = Path("data/processed/news_dataset.csv")


def load_dataset():
    print("Loading News Category Dataset...")
    df = pd.read_json(DATA_PATH, lines=True)
    print("Dataset Loaded Successfully!")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    return df


def explore_dataset(df):

    print("\nFirst Five Articles\n")
    print(df.head())

    print("\nColumn Names\n")
    print(df.columns.tolist())

    print("\nMissing Values\n")
    print(df.isnull().sum())


def select_columns(df):

    df = df[
        [
            "headline",
            "short_description",
            "date",
            "category",
            "link"
        ]
    ]

    return df


def rename_columns(df):

    df = df.rename(
        columns={
            "headline": "title",
            "short_description": "description",
            "link": "source"
        }
    )

    return df

def save_dataset(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to : {OUTPUT_PATH}")



if __name__ == "__main__":
    news_df = load_dataset()
    explore_dataset(news_df)
    news_df = select_columns(news_df)
    news_df = rename_columns(news_df)
    save_dataset(news_df)