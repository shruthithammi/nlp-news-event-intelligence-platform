import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI News Clustering & Timeline Builder",
    page_icon="📰",
    layout="wide",
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    labeled_df = pd.read_csv("outputs/clusters/labeled_news.csv")
    timeline_df = pd.read_csv("outputs/timelines/event_timelines.csv")
    summary_df = pd.read_csv("outputs/summaries/event_summaries.csv")

    # Dates
    labeled_df["date"] = pd.to_datetime(labeled_df["date"])
    timeline_df["start_date"] = pd.to_datetime(timeline_df["start_date"])
    timeline_df["end_date"] = pd.to_datetime(timeline_df["end_date"])

    # Remove spaces from event labels everywhere
    labeled_df["event_label"] = labeled_df["event_label"].astype(str).str.strip()
    timeline_df["event_label"] = timeline_df["event_label"].astype(str).str.strip()
    summary_df["event_label"] = summary_df["event_label"].astype(str).str.strip()

    # Summary column safety
    summary_df["summary"] = summary_df["summary"].fillna("Summary unavailable.")

    return labeled_df, timeline_df, summary_df


labeled_df, timeline_df, summary_df = load_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📰 AI News Clustering & Timeline Builder")
st.markdown(
    "### Semantic News Event Detection using **Sentence-BERT + HDBSCAN + DistilBART**"
)
st.markdown("---")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("🎯 Event Explorer")

events = sorted(
    labeled_df[labeled_df["cluster_id"] != -1]["event_label"].unique().tolist()
)

st.sidebar.write(f"**Available Events:** {len(events)}")

selected_event = st.sidebar.selectbox(
    "Select Event",
    ["All Events"] + events,
)

# --------------------------------------------------
# SEARCH
# --------------------------------------------------

st.subheader("🔍 Search News Articles")

search = st.text_input(
    "Search by news title, event name or AI summary",
    placeholder="Try: covid, ukraine, mariupol, biden, netflix, hong kong..."
)

filtered_df = labeled_df.copy()

# Event filter first
if selected_event != "All Events":
    filtered_df = filtered_df[
        filtered_df["event_label"] == selected_event
    ]

# Search filter
if search:
    matched_events = summary_df[
        summary_df["summary"].str.contains(search, case=False, na=False)
    ]["event_label"].unique()

    filtered_df = filtered_df[
        filtered_df["title"].str.contains(search, case=False, na=False)
        | filtered_df["event_label"].str.contains(search, case=False, na=False)
        | filtered_df["event_label"].isin(matched_events)
    ]

st.write(f"## Showing {len(filtered_df):,} Articles")

st.markdown("---")

# --------------------------------------------------
# METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📰 Total Articles", f"{len(filtered_df):,}")

with col2:
    st.metric(
        "🎯 Total Events",
        filtered_df["event_label"].nunique()
        if selected_event == "All Events"
        else 1,
    )

with col3:
    st.metric(
        "📅 Timeline Years",
        f"{timeline_df['start_date'].dt.year.min()} - {timeline_df['end_date'].dt.year.max()}",
    )

st.markdown("---")

# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

st.subheader("📊 Top Event Distribution")

pie_data = (
    filtered_df[filtered_df["cluster_id"] != -1]
    .groupby("event_label")
    .size()
    .reset_index(name="Articles")
    .sort_values("Articles", ascending=False)
    .head(10)
)

pie_chart = px.pie(
    pie_data,
    names="event_label",
    values="Articles",
    hole=0.45,
    title="Top 10 Biggest News Events",
)

st.plotly_chart(pie_chart, width="stretch")

st.markdown("---")

# --------------------------------------------------
# TIMELINE
# --------------------------------------------------

st.subheader("📅 Interactive Event Timeline")

timeline_display = timeline_df.copy()

if selected_event != "All Events":
    timeline_display = timeline_display[
        timeline_display["event_label"] == selected_event
    ]
else:
    timeline_display = (
        timeline_display.sort_values("total_articles", ascending=False).head(20)
    )

timeline_chart = px.timeline(
    timeline_display,
    x_start="start_date",
    x_end="end_date",
    y="event_label",
    color="total_articles",
    hover_data=["cluster_id", "total_articles"],
)

timeline_chart.update_yaxes(autorange="reversed")
timeline_chart.update_layout(height=650)

st.plotly_chart(timeline_chart, width="stretch")

st.markdown("---")

# --------------------------------------------------
# MONTHLY TREND
# --------------------------------------------------

st.subheader("📈 Monthly News Trend")

trend_df = filtered_df.copy()
trend_df["month"] = trend_df["date"].dt.to_period("M").astype(str)

monthly = (
    trend_df.groupby("month")
    .size()
    .reset_index(name="Articles")
)

trend_chart = px.line(
    monthly,
    x="month",
    y="Articles",
    markers=True,
    title="News Articles Published Per Month",
)

trend_chart.update_layout(height=450)

st.plotly_chart(trend_chart, width="stretch")

st.markdown("---")

# --------------------------------------------------
# AI SUMMARY
# --------------------------------------------------
# --------------------------------------------------
# AI EVENT SUMMARY
# --------------------------------------------------
# --------------------------------------------------
# AI EVENT SUMMARY (FIXED)
# --------------------------------------------------

st.subheader("🤖 AI Event Summary")

if selected_event != "All Events":

    # Get cluster_id of selected event
    selected_cluster = filtered_df["cluster_id"].iloc[0]

    # Match summary using cluster_id
    summary = summary_df[
        summary_df["cluster_id"] == selected_cluster
    ]

    if not summary.empty and pd.notna(summary.iloc[0]["summary"]):
        st.success(summary.iloc[0]["summary"])
    else:
        st.warning("AI summary is not available for this event.")

    # Event statistics
    article_count = len(filtered_df)

    start_date = filtered_df["date"].min().strftime("%d %b %Y")
    end_date = filtered_df["date"].max().strftime("%d %b %Y")

    c1, c2, c3 = st.columns(3)

    c1.metric("📰 Articles", article_count)
    c2.metric("📅 Start Date", start_date)
    c3.metric("📅 End Date", end_date)

else:
    st.info("Select an event from the sidebar to view its AI-generated summary.")
# --------------------------------------------------
# NEWS TABLE
# --------------------------------------------------

st.subheader("📰 Latest Clustered News Articles")

table_df = (
    filtered_df[["date", "title", "event_label"]]
    .sort_values("date", ascending=False)
    .head(20)
)

st.dataframe(
    table_df,
    width="stretch",
    hide_index=True,
)

# --------------------------------------------------
# DOWNLOAD CSV
# --------------------------------------------------

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Filtered News CSV",
    csv,
    file_name="filtered_news.csv",
    mime="text/csv",
)

st.markdown("---")

# --------------------------------------------------
# TOP EVENTS BAR
# --------------------------------------------------

st.subheader("📊 Top 10 Biggest Events")

bar_df = (
    filtered_df[filtered_df["cluster_id"] != -1]
    .groupby("event_label")
    .size()
    .reset_index(name="Articles")
    .sort_values("Articles", ascending=False)
    .head(10)
)

bar_chart = px.bar(
    bar_df,
    x="Articles",
    y="event_label",
    orientation="h",
    text="Articles",
    color="Articles",
)

bar_chart.update_layout(
    yaxis_title="",
    xaxis_title="Number of Articles",
    height=500,
)

st.plotly_chart(bar_chart, width="stretch")

st.markdown("---")

# --------------------------------------------------
# PROJECT WORKFLOW
# --------------------------------------------------

with st.expander("📚 How This AI Project Works"):

    st.markdown(
        """
### NLP Pipeline Used

**Step 1:** 📥 Load News Category Dataset

**Step 2:** 🧹 Clean Titles, Descriptions and Dates

**Step 3:** 🧠 Generate Sentence-BERT Embeddings

**Step 4:** 🎯 Cluster Similar News Articles using HDBSCAN

**Step 5:** 🏷️ Generate Event Labels using TF-IDF Keywords

**Step 6:** 📅 Build Chronological Event Timelines

**Step 7:** 🤖 Generate AI Event Summaries using DistilBART

**Step 8:** 📊 Visualize Everything using Streamlit + Plotly
"""
    )

st.markdown("---")

st.caption(
    "❤️ Built using Sentence-BERT, HDBSCAN, TF-IDF, DistilBART, Plotly and Streamlit."
)