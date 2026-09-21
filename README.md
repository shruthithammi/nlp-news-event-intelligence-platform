# 📰 AI-Powered Semantic News Event Intelligence Platform

An end-to-end NLP application that automatically groups thousands of news articles discussing the **same real-world event**, builds a **chronological timeline** for each event, and generates an **AI-powered summary** for quick understanding.

Built using **Sentence-BERT**, **HDBSCAN**, **TF-IDF**, **DistilBART**, **Plotly**, and **Streamlit**.

---

## 🚀 Project Overview

Reading hundreds of news articles about the same event is time-consuming and repetitive. This project solves that problem by identifying semantically similar news articles, clustering them into meaningful events, organizing those events chronologically, and producing concise AI-generated summaries.

Instead of manually reading multiple articles, users can explore a single event timeline and understand the story through one summarized view.

### Example Use Cases

* COVID-19 news timeline.
* Ukraine war / Mariupol event timeline.
* Hong Kong extradition protests.
* Netflix entertainment releases.
* Political and global news event tracking.

---

## ✨ Features

* Semantic clustering of similar news articles.
* Automatic event detection using HDBSCAN.
* AI-generated event labels using TF-IDF keywords.
* Chronological timeline builder for each event.
* AI-powered event summarization using DistilBART.
* Interactive Streamlit dashboard with search and filtering.
* Download filtered event articles as CSV.

---

## 🧠 NLP Pipeline

1. Load News Category Dataset.
2. Clean titles, descriptions, and publication dates.
3. Generate semantic embeddings using Sentence-BERT.
4. Cluster similar news articles using HDBSCAN.
5. Extract representative keywords using TF-IDF.
6. Build chronological timelines for detected events.
7. Generate AI summaries using DistilBART.
8. Visualize events in an interactive Streamlit dashboard.

---

## 🏗️ Project Architecture

User News Dataset
│
▼
Preprocessing
(Cleaning + Date Formatting)
│
▼
Sentence-BERT Embeddings
│
▼
HDBSCAN Clustering
│
▼
TF-IDF Event Label Generation
│
▼
Timeline Builder
│
▼
DistilBART Summarizer
│
▼
Streamlit Dashboard

---

## 🛠️ Tech Stack

| Category             | Technologies                                 |
| -------------------- | -------------------------------------------- |
| Programming Language | Python                                       |
| NLP Embeddings       | Sentence-BERT (`all-MiniLM-L6-v2`)           |
| Clustering Algorithm | HDBSCAN                                      |
| Keyword Extraction   | TF-IDF                                       |
| Summarization        | DistilBART (`sshleifer/distilbart-cnn-12-6`) |
| Data Processing      | Pandas, NumPy                                |
| Visualization        | Plotly                                       |
| Dashboard            | Streamlit                                    |
| Dataset              | Kaggle News Category Dataset                 |

---

## 📂 Project Structure

```text
nlp-news-event-intelligence-platform/
│
├── app.py                         # Streamlit Dashboard
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── clustering.py
│   ├── labeling.py
│   ├── timeline.py
│   └── summarizer.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── clusters/
│   ├── timelines/
│   └── summaries/
│
└── assets/                        # Dashboard screenshots (optional)
```

---

## 📊 Dataset

**Dataset:** Kaggle News Category Dataset

The dataset contains thousands of news articles collected across multiple years and categories.

> **Note:** Large dataset files are intentionally excluded from this repository using `.gitignore`. Download the dataset from Kaggle and place it inside `data/raw/` before running the project.

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shruthithammi/nlp-news-event-intelligence-platform.git
cd nlp-news-event-intelligence-platform
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the NLP Pipeline

Run each module sequentially.

```bash
python src/preprocessing.py
```

```bash
python src/embeddings.py
```

```bash
python src/clustering.py
```

```bash
python src/labeling.py
```

```bash
python src/timeline.py
```

```bash
python src/summarizer.py
```

---

## 🌐 Launch the Dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 🎯 Dashboard Features

* 📊 Top Event Distribution.
* 🔍 Semantic News Search.
* 📅 Interactive Timeline Visualization.
* 📈 Monthly News Publishing Trend.
* 🤖 AI Event Summary.
* 🔑 TF-IDF Event Keywords.
* 📰 Latest Clustered News Articles.
* 📥 Download Filtered Articles.

---

## 📸 Dashboard Preview

*Add screenshots inside the `assets/` folder and update the image paths below.*

<img width="1905" height="827" alt="image" src="https://github.com/user-attachments/assets/b5e71338-2c86-42b8-9cbd-14e267b8f7fd" />

<img width="1542" height="880" alt="image" src="https://github.com/user-attachments/assets/1b2a057b-dd6b-4737-b371-0f56a113bc8c" />

<img width="1916" height="1007" alt="image" src="https://github.com/user-attachments/assets/027d1ba6-09bc-4a3b-a69a-05e7284f8435" />



---

## 📈 Sample Event Output

**Event:** `coronavirus | covid | face`

* Event Cluster: COVID-19 Face Mask News
* Timeline: March 2020 – May 2022
* Articles Clustered: 22
* AI Summary Generated using DistilBART.

---

## 💡 Key Learnings

* Semantic text representation using transformer embeddings.
* Density-based clustering for event discovery.
* Keyword extraction using TF-IDF.
* Generative AI summarization for clustered documents.
* Interactive visualization using Streamlit and Plotly.

---

## 🔮 Future Improvements

* Retrieval-Augmented Generation (RAG) for event Q&A.
* Named Entity Recognition (NER) for people, organizations, and locations.
* Sentiment analysis for each news event.
* Interactive world map of news events.
* FAISS vector database for semantic search.
* Live NewsAPI integration for real-time event tracking.

---

## 👩‍💻 Author

**Shruthi Thammi**

AI & Machine Learning Engineer | Python | NLP | Generative AI | Streamlit | SQL | Power BI

* GitHub: https://github.com/shruthithammi
* LinkedIn: https://www.linkedin.com/in/shruthi-thammi/

---

## ⭐ If you found this project helpful

Give this repository a ⭐ on GitHub if you found it useful or inspiring.
