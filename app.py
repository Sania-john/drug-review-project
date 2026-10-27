import os
import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

# ============================================================
# DrugLens AI — Big Data Drug Review Analytics
# ============================================================
st.set_page_config(
    page_title="DrugLens AI | Intelligence Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Theme ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #080d1b 0%, #10172b 55%, #111827 100%);
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background: #0b1020;
        border-right: 1px solid #25334d;
    }
    h1, h2, h3 { color: #67e8f9 !important; }
    [data-testid="stMetric"] {
        background: #141f35;
        border: 1px solid #263d5c;
        padding: 16px;
        border-radius: 14px;
    }
    [data-testid="stMetricValue"] { color: #67e8f9; }
    .small-note { color: #94a3b8; font-size: 12px; letter-spacing: 1.5px; }
    .info-card {
        background: #101a2c;
        border: 1px solid #263d5c;
        border-radius: 12px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------- File locations ----------------------
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_CANDIDATES = [
    PROJECT_DIR / "drug_reviews_processed.csv",
    PROJECT_DIR / "powerbi_data" / "drug_reviews_processed.csv",
    PROJECT_DIR / "drug review dataset.csv",
    PROJECT_DIR / "drug_review_dataset.csv",
    PROJECT_DIR / "powerbi_data" / "drug review dataset.csv",
    PROJECT_DIR / "powerbi_data" / "drug_review_dataset.csv",
    Path.home() / "Downloads" / "drug_reviews_processed.csv",
    Path.home() / "Downloads" / "drug review dataset.csv",
]
DEFAULT_PBIX_CANDIDATES = [
    PROJECT_DIR / "bgdata.pbix",
    PROJECT_DIR / "drug_review.pbix",
    PROJECT_DIR / "powerbi_data" / "bgdata.pbix",
    Path.home() / "Downloads" / "bgdata.pbix",
    Path.home() / "Downloads" / "drug_review.pbix",
]
POWERBI_EMBED_URL = os.getenv("POWERBI_EMBED_URL", "").strip()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")


def first_existing(paths):
    for path in paths:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


# ---------------------- MongoDB connection -------------------
@st.cache_resource
def connect_mongodb(uri):
    client = MongoClient(uri, serverSelectionTimeoutMS=2500)
    client.admin.command("ping")
    return client, client["drug_db"]


mongo_client = None
db = None
mongo_error = None
try:
    mongo_client, db = connect_mongodb(MONGO_URI)
except Exception as exc:
    mongo_error = str(exc)


def load_collection(collection_name):
    """Load a MongoDB collection as a DataFrame; return empty on failure."""
    if db is None:
        return pd.DataFrame()
    try:
        records = list(db[collection_name].find())
        if not records:
            return pd.DataFrame()
        # Keep all user fields while dropping MongoDB's internal _id field.
        return pd.DataFrame(
            [{key: value for key, value in row.items() if key != "_id"}
             for row in records]
        )
    except Exception:
        return pd.DataFrame()


# ---------------------- Local CSV dataset --------------------
@st.cache_data(show_spinner="Loading drug review dataset...")
def load_raw_dataset(path_string, modified_time):
    # modified_time participates in the cache key so file updates reload.
    del modified_time
    frame = pd.read_csv(path_string, low_memory=False)
    # Accept both the original Drug Review dataset and the CSV exported from Colab.
    frame.columns = [str(column).strip() for column in frame.columns]
    lower_to_original = {column.lower(): column for column in frame.columns}
    aliases = {
        "drug": "drugName",
        "drugname": "drugName",
        "condition": "condition",
        "review": "review",
        "rating": "rating",
    }
    rename_map = {}
    for lower_name, target_name in aliases.items():
        if lower_name in lower_to_original and target_name not in frame.columns:
            rename_map[lower_to_original[lower_name]] = target_name
    if rename_map:
        frame = frame.rename(columns=rename_map)
    required = {"drugName", "condition", "review", "rating"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "CSV must contain drug/drugName, condition, review, and rating columns. "
            f"Found: {', '.join(map(str, frame.columns))}"
        )
    frame["drugName"] = frame["drugName"].fillna("").astype(str).str.strip()
    frame["condition"] = frame["condition"].fillna("").astype(str).str.strip()
    frame["review"] = frame["review"].fillna("").astype(str).map(html.unescape)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["rating"])
    frame = frame[
        (frame["drugName"] != "") &
        (frame["condition"] != "") &
        (frame["review"] != "")
    ].copy()
    frame["drug_lower"] = frame["drugName"].str.lower()
    frame["condition_lower"] = frame["condition"].str.lower()
    frame["review_lower"] = frame["review"].str.lower()
    return frame


dataset_path = first_existing(DEFAULT_DATASET_CANDIDATES)
raw_reviews = pd.DataFrame()
dataset_error = None
if dataset_path:
    try:
        raw_reviews = load_raw_dataset(
            str(dataset_path), dataset_path.stat().st_mtime
        )
    except Exception as exc:
        dataset_error = str(exc)

# ---------------------- Other collections --------------------
summary = load_collection("drug_summary")
mongo_effects = load_collection("side_effect_counts")

if not summary.empty:
    if "avg_rating" in summary.columns:
        summary["avg_rating"] = pd.to_numeric(summary["avg_rating"], errors="coerce")
    if "review_count" in summary.columns:
        summary["review_count"] = pd.to_numeric(summary["review_count"], errors="coerce")
    for col in ("condition", "drug"):
        if col in summary.columns:
            summary[col] = summary[col].fillna("").astype(str).str.strip()
    if not {"condition", "drug", "avg_rating"}.issubset(summary.columns):
        summary = pd.DataFrame()

if not mongo_effects.empty:
    if "_id" in mongo_effects.columns and "term" not in mongo_effects.columns:
        mongo_effects = mongo_effects.rename(columns={"_id": "term"})
    if "value" in mongo_effects.columns and "mentions" not in mongo_effects.columns:
        mongo_effects = mongo_effects.rename(columns={"value": "mentions"})
    if {"term", "mentions"}.issubset(mongo_effects.columns):
        mongo_effects["mentions"] = pd.to_numeric(
            mongo_effects["mentions"], errors="coerce"
        )
        mongo_effects = mongo_effects.dropna(subset=["mentions"])
    else:
        mongo_effects = pd.DataFrame()


# ---------------- Side-effect keyword analysis ----------------
SIDE_EFFECT_TERMS = [
    "nausea", "headache", "dizziness", "fatigue", "insomnia",
    "drowsiness", "rash", "weight gain", "dry mouth", "anxiety",
    "vomiting", "diarrhea", "constipation", "sweating", "depression",
    "pain", "itching", "cramps",
]


@st.cache_data(show_spinner="Calculating review keyword mentions...")
def calculate_side_effect_comparison(frame, terms):
    if frame.empty:
        return pd.DataFrame()
    rows = []
    total_reviews = len(frame)
    low = frame[frame["rating"] <= 4]
    for term in terms:
        all_mentions = frame["review_lower"].str.contains(
            term, regex=False, na=False
        )
        low_mentions = low["review_lower"].str.contains(
            term, regex=False, na=False
        )
        rows.append({
            "term": term,
            "all_reviews_mentions": int(all_mentions.sum()),
            "all_reviews_pct": round(100 * all_mentions.sum() / total_reviews, 3)
                if total_reviews else 0,
            "low_rated_mentions": int(low_mentions.sum()),
            "low_rated_pct": round(100 * low_mentions.sum() / len(low), 3)
                if len(low) else 0,
            "all_review_count": total_reviews,
            "low_rated_review_count": len(low),
        })
    return pd.DataFrame(rows).sort_values(
        "all_reviews_mentions", ascending=False
    )


side_effect_comparison = calculate_side_effect_comparison(
    raw_reviews, SIDE_EFFECT_TERMS
)

# Prefer the user's existing Power BI comparison export when available.
# Expected columns: term, all_reviews_pct, low_rated_pct.
# Extra count columns are retained if the CSV includes them.
SIDE_EFFECT_COMPARE_CANDIDATES = [
    PROJECT_DIR / "powerbi_data" / "side_effects_compare.csv",
    PROJECT_DIR / "side_effects_compare.csv",
]
side_effect_compare_path = first_existing(SIDE_EFFECT_COMPARE_CANDIDATES)
if side_effect_compare_path:
    try:
        powerbi_compare = pd.read_csv(side_effect_compare_path)
        powerbi_compare.columns = [
            str(column).strip().lower() for column in powerbi_compare.columns
        ]
        # Accommodate common alternative column labels from exported reports.
        rename_map = {}
        if "side_effect" in powerbi_compare.columns and "term" not in powerbi_compare.columns:
            rename_map["side_effect"] = "term"
        if "all_reviews_percentage" in powerbi_compare.columns and "all_reviews_pct" not in powerbi_compare.columns:
            rename_map["all_reviews_percentage"] = "all_reviews_pct"
        if "low_rated_percentage" in powerbi_compare.columns and "low_rated_pct" not in powerbi_compare.columns:
            rename_map["low_rated_percentage"] = "low_rated_pct"
        powerbi_compare = powerbi_compare.rename(columns=rename_map)

        if {"term", "all_reviews_pct", "low_rated_pct"}.issubset(powerbi_compare.columns):
            powerbi_compare["term"] = powerbi_compare["term"].astype(str)
            for column in ["all_reviews_pct", "low_rated_pct",
                           "all_reviews_mentions", "low_rated_mentions"]:
                if column in powerbi_compare.columns:
                    powerbi_compare[column] = pd.to_numeric(
                        powerbi_compare[column], errors="coerce"
                    )
            side_effect_comparison = powerbi_compare
    except Exception:
        # Keep the locally calculated comparison if the optional export is malformed.
        pass


# ---------------------- Local Spark outputs -------------------
SPARK_OUTPUT_DIR = PROJECT_DIR / "spark_output"
SPARK_OUTPUT_CANDIDATES = {
    "drug_summary": [
        SPARK_OUTPUT_DIR / "drug_summary.csv",
        PROJECT_DIR / "drug_summary.csv",
    ],
    "side_effect_counts": [
        SPARK_OUTPUT_DIR / "side_effect_counts.csv",
        PROJECT_DIR / "side_effect_counts.csv",
    ],
    "low_rating_side_effects": [
        SPARK_OUTPUT_DIR / "low_rating_side_effects.csv",
        PROJECT_DIR / "low_rating_side_effects.csv",
    ],
    "drug_side_effects": [
        SPARK_OUTPUT_DIR / "drug_side_effects.csv",
        PROJECT_DIR / "drug_side_effects.csv",
    ],
}


@st.cache_data(show_spinner="Loading Spark analytics outputs...")
def load_spark_csv(path_string, modified_time):
    del modified_time
    return pd.read_csv(path_string, low_memory=False)


def get_spark_output(key):
    path = first_existing(SPARK_OUTPUT_CANDIDATES[key])
    if path is None:
        return pd.DataFrame(), None
    try:
        frame = load_spark_csv(str(path), path.stat().st_mtime)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        return frame, path
    except Exception as exc:
        return pd.DataFrame(), f"ERROR: {exc}"


spark_drug_summary, spark_summary_path = get_spark_output("drug_summary")
spark_effect_counts, spark_effect_path = get_spark_output("side_effect_counts")
spark_low_effects, spark_low_effect_path = get_spark_output("low_rating_side_effects")
spark_drug_effects, spark_drug_effect_path = get_spark_output("drug_side_effects")


def normalize_spark_columns():
    # Coerce numeric output columns and normalize text columns for reliable charts.
    for frame, numeric_cols, text_cols in [
        (spark_drug_summary, ["avg_rating", "review_count"], ["condition", "drug"]),
        (spark_effect_counts, ["reviews", "mentions", "count"], ["term"]),
        (spark_low_effects, ["reviews", "mentions", "count"], ["term"]),
        (spark_drug_effects, ["reviews", "mentions", "count"], ["drug", "term"]),
    ]:
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in text_cols:
            if col in frame.columns:
                frame[col] = frame[col].fillna("").astype(str).str.strip()


normalize_spark_columns()

# ----------------------- Sidebar navigation -------------------
st.sidebar.markdown("# 🧬 DrugLens AI")
st.sidebar.caption("BIG DATA INTELLIGENCE PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Spark Processing",
        "Drug Explorer",
        "Side-Effect Intelligence",
        "Power BI Dashboard",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### SYSTEM STATUS")
if db is not None:
    st.sidebar.success("MongoDB connected")
else:
    st.sidebar.warning("MongoDB unavailable")

if dataset_path:
    st.sidebar.success("Review CSV found")
else:
    st.sidebar.warning("Review CSV not found")

st.sidebar.caption("Google Colab / Spark • Python • Pandas • MongoDB • Power BI")

# ---------------------------- Header --------------------------
st.markdown(
    "<p class='small-note'>PHARMACEUTICAL ANALYTICS / DATA INTELLIGENCE</p>",
    unsafe_allow_html=True,
)
st.title("DrugLens AI")
st.markdown("Explore drug reviews. Discover patterns. Understand feedback.")
st.divider()

# ---------------------- Spark Processing ----------------------
if page == "Spark Processing":
    st.subheader("⚡ Local PySpark Analytics")
    st.markdown(
        "This page displays the CSV outputs generated by `spark_jobs.py`. "
        "Run the Spark job first; the dashboard reads the saved results from `spark_output/`."
    )

    spark_frames = [
        ("Drug summary", spark_drug_summary, spark_summary_path),
        ("Side-effect counts", spark_effect_counts, spark_effect_path),
        ("Low-rating side effects", spark_low_effects, spark_low_effect_path),
        ("Drug / side-effect pairs", spark_drug_effects, spark_drug_effect_path),
    ]
    present_count = sum(not frame.empty for _, frame, _ in spark_frames)
    if present_count:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spark output tables", f"{present_count}/4")
        m2.metric("Drug-condition pairs", f"{len(spark_drug_summary):,}")
        m3.metric(
            "Side-effect terms",
            f"{spark_effect_counts['term'].nunique():,}" if "term" in spark_effect_counts.columns else "N/A",
        )
        m4.metric("Drug-effect pairs", f"{len(spark_drug_effects):,}")
        st.caption(f"Spark output directory: `{SPARK_OUTPUT_DIR}`")

        if not spark_drug_summary.empty:
            st.markdown("### ⭐ Drug rating summary")
            if {"drug", "condition", "avg_rating"}.issubset(spark_drug_summary.columns):
                summary_view = spark_drug_summary.copy()
                condition_values = sorted(summary_view["condition"].replace("", pd.NA).dropna().unique().tolist())
                selected_spark_condition = st.selectbox(
                    "Filter Spark summary by condition",
                    ["All conditions"] + condition_values,
                    key="spark_condition_filter",
                )
                if selected_spark_condition != "All conditions":
                    summary_view = summary_view[summary_view["condition"] == selected_spark_condition]
                if not summary_view.empty:
                    top_spark = summary_view.nlargest(10, "avg_rating")
                    fig = px.bar(
                        top_spark.sort_values("avg_rating"),
                        x="avg_rating", y="drug", orientation="h",
                        color="avg_rating",
                        hover_data=[c for c in ["condition", "review_count"] if c in top_spark.columns],
                        title="Top 10 Average-Rated Drug Entries",
                        labels={"avg_rating": "Average rating", "drug": "Drug"},
                        color_continuous_scale=["#7c3aed", "#22d3ee"],
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(summary_view, use_container_width=True)
            else:
                st.warning("drug_summary.csv must contain condition, drug, and avg_rating columns.")

        if not spark_effect_counts.empty and "term" in spark_effect_counts.columns:
            st.markdown("### 🧪 Most-mentioned side-effect terms")
            effect_count_col = next((c for c in ["reviews", "mentions", "count"] if c in spark_effect_counts.columns), None)
            if effect_count_col:
                top_effects = spark_effect_counts.nlargest(15, effect_count_col).sort_values(effect_count_col)
                fig = px.bar(
                    top_effects, x=effect_count_col, y="term", orientation="h",
                    title="Top Side-Effect Keyword Mentions (Spark)",
                    labels={effect_count_col: "Reviews / mentions", "term": "Term"},
                    color=effect_count_col, color_continuous_scale=["#7c3aed", "#22d3ee"],
                )
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(spark_effect_counts, use_container_width=True)

        if not spark_low_effects.empty and "term" in spark_low_effects.columns:
            st.markdown("### ⚠️ Side effects in low-rated reviews (rating ≤ 4)")
            low_count_col = next((c for c in ["reviews", "mentions", "count"] if c in spark_low_effects.columns), None)
            if low_count_col:
                low_view = spark_low_effects.nlargest(15, low_count_col).sort_values(low_count_col)
                fig = px.bar(
                    low_view, x=low_count_col, y="term", orientation="h",
                    title="Most-mentioned Terms in Low-rated Reviews",
                    labels={low_count_col: "Low-rated reviews / mentions", "term": "Term"},
                    color=low_count_col, color_continuous_scale=["#be123c", "#f9a8d4"],
                )
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(spark_low_effects, use_container_width=True)

        if not spark_drug_effects.empty:
            st.markdown("### 🔬 Drug-specific side-effect terms")
            if {"drug", "term"}.issubset(spark_drug_effects.columns):
                drug_names = sorted(spark_drug_effects["drug"].replace("", pd.NA).dropna().unique().tolist())
                selected_spark_drug = st.selectbox(
                    "Select a drug to inspect keyword mentions",
                    ["All drugs"] + drug_names,
                    key="spark_drug_effect_filter",
                )
                drug_effect_view = spark_drug_effects.copy()
                if selected_spark_drug != "All drugs":
                    drug_effect_view = drug_effect_view[drug_effect_view["drug"] == selected_spark_drug]
                count_col = next((c for c in ["reviews", "mentions", "count"] if c in drug_effect_view.columns), None)
                if count_col and not drug_effect_view.empty:
                    plot_view = drug_effect_view.nlargest(15, count_col).sort_values(count_col)
                    fig = px.bar(
                        plot_view, x=count_col, y="term", orientation="h",
                        title="Side-effect keywords by drug",
                        labels={count_col: "Reviews / mentions", "term": "Term"},
                        color=count_col, color_continuous_scale=["#2563eb", "#22d3ee"],
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(drug_effect_view, use_container_width=True)

        for label, frame, path in spark_frames:
            if isinstance(path, str) and path.startswith("ERROR:"):
                st.error(f"{label}: {path}")

        st.download_button(
            "Download Spark drug summary",
            data=spark_drug_summary.to_csv(index=False).encode("utf-8"),
            file_name="drug_summary.csv",
            mime="text/csv",
            disabled=spark_drug_summary.empty,
        )
    else:
        st.warning("No Spark output CSVs were found yet.")
        st.code(
            '$env:JAVA_HOME = "$env:LOCALAPPDATA\\Programs\\Eclipse Adoptium\\jdk-21.0.12.101-hotspot"\n'
            '$env:Path = "$env:JAVA_HOME\\bin;$env:Path"\n'
            '$env:PYTHONHASHSEED = "0"\n'
            "python spark_jobs.py",
            language="powershell",
        )
        st.info(
            "After the job completes, check that `spark_output` contains "
            "`drug_summary.csv`, `side_effect_counts.csv`, "
            "`low_rating_side_effects.csv`, and `drug_side_effects.csv`, "
            "then refresh this Streamlit page."
        )

# -------------------------- Overview --------------------------
if page == "Overview":
    st.subheader("📊 Analytics Overview")

    if not summary.empty:
        total_pairs = len(summary)
        total_summary_reviews = (
            int(summary["review_count"].fillna(0).sum())
            if "review_count" in summary.columns else 0
        )
        avg_rating = summary["avg_rating"].mean()
        conditions_count = summary["condition"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Drug-condition pairs", f"{total_pairs:,}")
        c2.metric("Reviews represented", f"{total_summary_reviews:,}")
        c3.metric("Conditions", f"{conditions_count:,}")
        c4.metric(
            "Mean average rating",
            f"{avg_rating:.2f} / 10" if pd.notna(avg_rating) else "N/A",
        )

        st.subheader("⚡ Rating Intelligence")
        condition_options = sorted(
            summary["condition"].replace("", pd.NA).dropna().unique().tolist()
        )
        selected_condition = st.selectbox(
            "Filter by condition", ["All conditions"] + condition_options
        )
        view = summary.copy()
        if selected_condition != "All conditions":
            view = view[view["condition"] == selected_condition]

        if not view.empty:
            top = view.nlargest(10, "avg_rating").copy()
            fig = px.bar(
                top.sort_values("avg_rating"),
                x="avg_rating",
                y="drug",
                color="avg_rating",
                orientation="h",
                hover_data=["condition", "review_count"]
                if "review_count" in top.columns else ["condition"],
                color_continuous_scale=["#2563eb", "#22d3ee"],
                title="Top 10 Highest-Rated Drug Entries",
                labels={"avg_rating": "Average rating", "drug": "Drug"},
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No records match the selected condition.")
    elif not raw_reviews.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Clean review records", f"{len(raw_reviews):,}")
        c2.metric("Unique drugs", f"{raw_reviews['drugName'].nunique():,}")
        c3.metric("Conditions", f"{raw_reviews['condition'].nunique():,}")
        st.info(
            "The local CSV is loaded. Run your MongoDB processing scripts "
            "to populate the drug_summary collection for rating summaries."
        )
    else:
        st.warning(
            "No summary data or source CSV was found. Put 'drug review dataset.csv' "
            "in the same folder as app.py, or start MongoDB and populate drug_db."
        )

    if dataset_path:
        st.caption(f"Source CSV: {dataset_path}")

# ------------------------- Drug Explorer ----------------------
elif page == "Drug Explorer":
    st.subheader("🔎 Drug Explorer")

    if not summary.empty:
        drug_options = sorted(
            summary["drug"].replace("", pd.NA).dropna().unique().tolist()
        )
        if drug_options:
            selected_drug = st.selectbox("Search or select a drug", drug_options)
            result = summary[summary["drug"] == selected_drug].copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Conditions listed", f"{result['condition'].nunique():,}")
            mean_rating = result["avg_rating"].mean()
            c2.metric(
                "Mean rating",
                f"{mean_rating:.2f} / 10" if pd.notna(mean_rating) else "N/A",
            )
            if "review_count" in result.columns:
                c3.metric(
                    "Reviews represented",
                    f"{int(result['review_count'].fillna(0).sum()):,}",
                )

            fig = px.bar(
                result.sort_values("avg_rating"),
                x="avg_rating",
                y="condition",
                color="avg_rating",
                orientation="h",
                color_continuous_scale=["#7c3aed", "#22d3ee"],
                title="Average Rating by Condition",
                labels={"avg_rating": "Average rating", "condition": "Condition"},
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(result, use_container_width=True)
        else:
            st.info("No drug names are available in drug_summary.")

    elif not raw_reviews.empty:
        drug_options = sorted(raw_reviews["drugName"].unique().tolist())
        selected_drug = st.selectbox("Search or select a drug", drug_options)
        result = raw_reviews[raw_reviews["drugName"] == selected_drug]
        c1, c2 = st.columns(2)
        c1.metric("Review records", f"{len(result):,}")
        c2.metric("Mean rating", f"{result['rating'].mean():.2f} / 10")
        st.dataframe(
            result[[c for c in ["condition", "rating", "date", "usefulCount", "review"]
                    if c in result.columns]].head(100),
            use_container_width=True,
        )
    else:
        st.warning("Drug data is unavailable. Add the source CSV or populate MongoDB.")

# ------------------- Side-Effect Intelligence -----------------
elif page == "Side-Effect Intelligence":
    st.subheader("⚡ Side-Effect Intelligence")
    st.markdown(
        "Compare how often tracked terms appear in all reviews and in low-rated "
        "reviews (rating ≤ 4). These are text mentions, not verified clinical rates."
    )

    # Prefer local source CSV for a meaningful comparison. It can calculate
    # both all-review and low-rated percentages from the original review text.
    if not side_effect_comparison.empty:
        a, b, c = st.columns(3)
        if {"all_review_count", "low_rated_review_count"}.issubset(side_effect_comparison.columns):
            total = int(side_effect_comparison["all_review_count"].iloc[0])
            low_total = int(side_effect_comparison["low_rated_review_count"].iloc[0])
            a.metric("Reviews analyzed", f"{total:,}")
            b.metric("Low-rated reviews (≤ 4)", f"{low_total:,}")
        else:
            a.metric("Comparison source", "Power BI CSV" if side_effect_compare_path else "CSV analysis")
            b.metric("Percentage metrics", "Available")
        c.metric("Tracked terms", f"{len(side_effect_comparison):,}")

        if side_effect_compare_path:
            st.caption(f"Using existing Power BI export: {side_effect_compare_path}")

        st.divider()
        search_term = st.text_input(
            "Search a side-effect term",
            placeholder="e.g. headache, nausea, dizziness",
        )
        view = side_effect_comparison.copy()
        if search_term.strip():
            view = view[
                view["term"].str.contains(search_term.strip(), case=False, regex=False)
            ]

        metric_choice = st.radio(
            "Compare by",
            ["Percentage of reviews", "Number of mentions"],
            horizontal=True,
        )
        if metric_choice == "Percentage of reviews":
            chart_view = view.melt(
                id_vars=["term"],
                value_vars=["all_reviews_pct", "low_rated_pct"],
                var_name="review_group",
                value_name="percentage",
            )
            chart_view["review_group"] = chart_view["review_group"].map({
                "all_reviews_pct": "All reviews",
                "low_rated_pct": "Low-rated reviews (≤ 4)",
            })
            fig = px.bar(
                chart_view,
                x="term",
                y="percentage",
                color="review_group",
                barmode="group",
                title="Side-Effect Term Prevalence: All vs Low-Rated Reviews",
                labels={"term": "Tracked term", "percentage": "Reviews mentioning term (%)",
                        "review_group": "Review group"},
                color_discrete_sequence=["#22d3ee", "#a78bfa"],
            )
        else:
            if {"all_reviews_mentions", "low_rated_mentions"}.issubset(view.columns):
                count_columns = ["all_reviews_mentions", "low_rated_mentions"]
            elif {"all_reviews_count", "low_rated_count"}.issubset(view.columns):
                count_columns = ["all_reviews_count", "low_rated_count"]
            else:
                count_columns = []

            if count_columns:
                chart_view = view.melt(
                    id_vars=["term"],
                    value_vars=count_columns,
                    var_name="review_group",
                    value_name="mentions",
                )
                chart_view["review_group"] = chart_view["review_group"].map({
                    count_columns[0]: "All reviews",
                    count_columns[1]: "Low-rated reviews (≤ 4)",
                })
                fig = px.bar(
                    chart_view,
                    x="term",
                    y="mentions",
                    color="review_group",
                    barmode="group",
                    title="Side-Effect Keyword Mentions: All vs Low-Rated Reviews",
                    labels={"term": "Tracked term", "mentions": "Reviews mentioning term",
                            "review_group": "Review group"},
                    color_discrete_sequence=["#22d3ee", "#a78bfa"],
                )
            else:
                st.info(
                    "The selected Power BI CSV contains percentages but no mention counts. "
                    "Choose 'Percentage of reviews' to chart the available fields."
                )
                st.stop()

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-35,
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True)

        st.download_button(
            "Download comparison CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="side_effects_compare.csv",
            mime="text/csv",
        )
        st.caption(
            "A review is counted once per term if the term occurs anywhere in its text. "
            "Terms can appear in positive, negative, or negated statements; the chart "
            "does not determine whether a medicine caused a symptom."
        )

    elif not mongo_effects.empty:
        st.info(
            "The source review CSV was not found, so this page is showing MongoDB "
            "keyword counts only. The comparison chart needs the original review CSV."
        )
        st.metric("Tracked terms", f"{mongo_effects['term'].nunique():,}")
        fig = px.bar(
            mongo_effects.nlargest(15, "mentions").sort_values("mentions"),
            x="mentions",
            y="term",
            orientation="h",
            color="mentions",
            color_continuous_scale=["#7c3aed", "#22d3ee"],
            title="Most Mentioned Side-Effect Terms",
            labels={"mentions": "Keyword mentions", "term": "Term"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mongo_effects, use_container_width=True)
    else:
        st.warning(
            "Side-effect analysis needs 'drug review dataset.csv' beside app.py "
            "or the side_effect_counts collection in MongoDB."
        )
        if dataset_error:
            st.error(dataset_error)

# ---------------------- Power BI Dashboard --------------------
elif page == "Power BI Dashboard":
    st.subheader("📈 Power BI Dashboard")
    st.markdown(
        "Your Power BI report is `bgdata.pbix`. A local `.pbix` file cannot be "
        "rendered directly inside Streamlit; open it in Power BI Desktop, or use "
        "an authorized embed URL to display it inside this page."
    )

    pbix_path = first_existing(DEFAULT_PBIX_CANDIDATES)

    if POWERBI_EMBED_URL:
        st.success("Power BI embed URL detected.")
        st.components.v1.iframe(
            POWERBI_EMBED_URL,
            height=760,
            scrolling=True,
        )
        st.caption(
            "Make sure the embed method is authorized for your audience and data. "
            "Do not publish private report data publicly."
        )
    else:
        st.info("No Power BI embed URL is configured.")
        if pbix_path:
            st.success(f"Power BI file found: {pbix_path}")
            try:
                with open(pbix_path, "rb") as file_handle:
                    st.download_button(
                        "Download / copy Power BI report",
                        data=file_handle.read(),
                        file_name=pbix_path.name,
                        mime="application/octet-stream",
                    )
            except OSError as exc:
                st.warning(f"Could not read the PBIX file: {exc}")

            st.markdown("**To open the dashboard:**")
            st.code(str(pbix_path), language="text")
            st.markdown(
                "1. Open Power BI Desktop.\n"
                "2. Select **File → Open report**.\n"
                "3. Choose `bgdata.pbix` from the path shown above.\n"
                "4. Use your existing report pages: Best Drug by Condition, "
                "Side-Effect Analysis, and Drug Lookup."
            )
        else:
            st.warning(
                "Could not locate bgdata.pbix automatically. Copy it into the "
                "same folder as app.py, or into the Downloads folder for the "
                "Windows account running Streamlit."
            )
            st.caption(
                "Expected locations include the project folder and "
                "~/Downloads/bgdata.pbix."
            )

        st.markdown("#### Optional: embed a published report")
        st.markdown(
            "If you have a suitable Power BI embed URL, set the environment "
            "variable before starting Streamlit. Example for PowerShell:"
        )
        st.code(
            '$env:POWERBI_EMBED_URL="YOUR_AUTHORIZED_EMBED_URL"\n'
            "streamlit run app.py",
            language="powershell",
        )
        st.caption(
            "Do not put passwords, access tokens, or private credentials in app.py. "
            "A public Publish to web link can expose the report to anyone."
        )

    st.divider()
    st.markdown("#### Power BI data files")
    expected_csvs = [
        PROJECT_DIR / "powerbi_data" / "drug_summary.csv",
        PROJECT_DIR / "powerbi_data" / "side_effects_compare.csv",
        PROJECT_DIR / "powerbi_data" / "drug_side_effects.csv",
        PROJECT_DIR / "drug_summary.csv",
        PROJECT_DIR / "side_effects_compare.csv",
        PROJECT_DIR / "drug_side_effects.csv",
    ]
    found_csvs = []
    seen = set()
    for path in expected_csvs:
        if path.is_file() and str(path.resolve()) not in seen:
            found_csvs.append(path)
            seen.add(str(path.resolve()))

    if found_csvs:
        for path in found_csvs:
            st.write(f"✅ `{path}`")
    else:
        st.write(
            "No Power BI CSV exports found beside app.py or in powerbi_data/. "
            "The app calculates the side-effect comparison directly from the "
            "source review CSV when available."
        )

# --------------------------- About ----------------------------
elif page == "About Project":
    st.subheader("🧠 About DrugLens AI")
    st.markdown("""
    **Project title:** DrugLens AI — Big Data Analytics of Drug Reviews

    **Objective:** Process drug-review data, summarize ratings by drug and
    condition, and identify frequently mentioned terms in review text.

    **Technology stack**
    - Python and Pandas — data cleaning and preparation
    - MongoDB — document storage and aggregation
    - MapReduce / aggregation pipelines — data processing
    - Streamlit and Plotly — interactive application and visualizations
    - Power BI — business intelligence reporting

    **Data workflow**

    `CSV → Cleaning → MongoDB → MapReduce / Aggregation → Analytics → Dashboard`

    **Interpretation note:** User ratings and review text reflect reported
    experiences. Keyword counts are not verified clinical side-effect rates
    and do not establish that a medicine caused a symptom.
    """)

# ---------------------------- Footer -------------------------
st.divider()
st.markdown(
    "<p class='small-note' style='text-align:center;'>"
    "DRUGLENS AI • BIG DATA INTELLIGENCE • ACADEMIC PROJECT"
    "</p>",
    unsafe_allow_html=True,
)
