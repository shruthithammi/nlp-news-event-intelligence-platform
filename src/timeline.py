#Arrange clustered news articles chronologically.

import pandas as pd
from pathlib import Path
import plotly.express as px

INPUT_PATH = Path("outputs/clusters/labeled_news.csv")
OUTPUT_PATH = Path("outputs/timelines/event_timelines.csv")

def load_dataset():
    print("Loading labeled dataset...")
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

def remove_noise(df):
    print("Removing noise articles...")
    df = df[df["cluster_id"] != -1]
    print(f"Remaining Articles : {len(df)}")
    return df

def sort_by_date(df):
    print("Sorting articles chronologically...")
    df = df.sort_values(
        by=["cluster_id", "date"]
    )
    return df


def create_timelines(df):
    timeline_data = []
    clusters = sorted(df["cluster_id"].unique())
    for cluster in clusters:
        cluster_df = df[df["cluster_id"] == cluster].sort_values("date")
        event_name = cluster_df["event_label"].iloc[0]
        start_date = cluster_df["date"].min()
        end_date = cluster_df["date"].max()
        total_articles = len(cluster_df)

        # First 3 headlines
        top_articles = cluster_df["title"].head(3).tolist()

        timeline_data.append({
            "cluster_id": cluster,
            "event_label": event_name,
            "start_date": start_date,
            "end_date": end_date,
            "total_articles": total_articles,
            "sample_articles": " || ".join(top_articles)
        })

    timeline_df = pd.DataFrame(timeline_data)

    timeline_df = timeline_df.sort_values("start_date")

    return timeline_df

def create_timeline_plot(timeline_df):

    fig = px.timeline(
        timeline_df,
        x_start="start_date",
        x_end="end_date",
        y="event_label",
        color="total_articles",
        hover_data=["cluster_id", "total_articles"],
        title="News Event Timeline"
    )

    fig.update_yaxes(autorange="reversed")

    fig.write_html("outputs/plots/news_event_timeline.html")

    print("Timeline visualization saved to outputs/plots/news_event_timeline.html")

def save_timeline(df):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nTimeline saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    news_df = load_dataset()
    news_df = remove_noise(news_df)
    news_df = sort_by_date(news_df)
    timeline_df = create_timelines(news_df)
    print("\nFirst 10 Timeline Events\n")
    print(timeline_df.head(10))
    save_timeline(timeline_df)
    print("\nTimeline Builder Completed Successfully!")