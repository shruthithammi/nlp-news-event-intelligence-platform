import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("Loading labeled dataset...")

df = pd.read_csv("outputs/clusters/labeled_news.csv")

# Remove noise cluster
df = df[df["cluster_id"] != -1].copy()

print("Total clustered articles :", len(df))

# --------------------------------------------------
# Load DistilBART Model
# --------------------------------------------------

print("\nLoading AI Summarization Model...")

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Model Loaded Successfully! Using {device.upper()}")

# --------------------------------------------------
# Generate Summary Function
# --------------------------------------------------

def generate_summary(text):

    if pd.isna(text) or text.strip() == "":
        return "Summary not available."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=60,
        min_length=20,
        do_sample=False
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

# --------------------------------------------------
# Summarize Each Event
# --------------------------------------------------

summaries = []

clusters = sorted(df["cluster_id"].unique())

print("\nSummarizing Events...")

for cluster in tqdm(clusters):

    event_df = df[df["cluster_id"] == cluster]

    event_label = event_df["event_label"].iloc[0]

    # Combine first 5 titles + descriptions
    combined_text = " ".join(
        (
            event_df["title"].fillna("")
            + ". "
            + event_df["description"].fillna("")
        ).head(5).tolist()
    )

    summary = generate_summary(combined_text)

    summaries.append({
        "cluster_id": cluster,
        "event_label": event_label,
        "summary": summary
    })

# --------------------------------------------------
# Save All Summaries
# --------------------------------------------------

summary_df = pd.DataFrame(summaries)

os.makedirs("outputs/summaries", exist_ok=True)

summary_df.to_csv(
    "outputs/summaries/event_summaries.csv",
    index=False
)

print("\nFirst 10 Event Summaries\n")
print(summary_df.head(10))

print("\nTotal summaries generated:", len(summary_df))
print("Saved summaries to outputs/summaries/event_summaries.csv")
print("\nSummarizer Module Completed Successfully!")