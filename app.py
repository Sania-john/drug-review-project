import streamlit as st
import pandas as pd
from pymongo import MongoClient

st.set_page_config(page_title="Drug Review Analyzer", layout="wide")
db = MongoClient("mongodb://localhost:27017")["drug_db"]


@st.cache_data
def load_summary():
    return pd.DataFrame(list(db.drug_summary.find({}, {"_id": 0})))


@st.cache_data
def load_side_effect_compare():
    all_total = db.reviews.count_documents({})
    low_total = db.reviews.count_documents({"rating": {"$lte": 4}})
    allr = {d["_id"]: d["value"] / all_total * 100 for d in db.side_effect_counts.find()}
    low = {d["_id"]: d["value"] / low_total * 100 for d in db.side_effect_low_rated.find()}
    rows = [{"term": t, "All reviews %": round(allr.get(t, 0), 2),
             "Low-rated %": round(low.get(t, 0), 2)} for t in allr]
    return pd.DataFrame(rows).sort_values("All reviews %", ascending=False)


summary = load_summary()

st.title("Drug Review Effectiveness and Side-Effect Analyzer")
st.caption(f"{db.reviews.count_documents({}):,} reviews stored in MongoDB | "
           "MapReduce results from collections in drug_db")

tab1, tab2, tab3 = st.tabs(["Best drug by condition", "Side effects", "Drug lookup"])

# ---------------- Tab 1 ----------------
with tab1:
    min_reviews = st.slider("Minimum reviews per drug", 1, 100, 10)
    filtered = summary[summary["review_count"] >= min_reviews]
    conditions = (filtered.groupby("condition")["review_count"].sum()
                  .sort_values(ascending=False).index.tolist())
    cond = st.selectbox("Condition (sorted by number of reviews)", conditions)
    sub = (filtered[filtered["condition"] == cond]
           .sort_values("avg_rating", ascending=False).head(10))
    st.subheader(f"Top drugs for {cond}")
    st.dataframe(sub[["drug", "avg_rating", "review_count"]], hide_index=True,
                 use_container_width=True)
    st.bar_chart(sub.set_index("drug")["avg_rating"])

# ---------------- Tab 2 ----------------
with tab2:
    cmp_df = load_side_effect_compare()
    st.subheader("Side-effect terms: all reviews vs low-rated reviews (rating 4 or below)")
    st.bar_chart(cmp_df.set_index("term")[["All reviews %", "Low-rated %"]])
    st.caption("Terms that appear in the drug's own condition name are excluded. "
               "Keyword matching misses misspellings and negation.")
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)

# ---------------- Tab 3 ----------------
with tab3:
    totals = summary.groupby("drug")["review_count"].sum()
    drugs = totals[totals >= 50].sort_values(ascending=False).index.tolist()
    drug = st.selectbox("Drug (with at least 50 reviews)", drugs)

    n = db.reviews.count_documents({"drugName": drug})
    avg = list(db.reviews.aggregate([
        {"$match": {"drugName": drug}},
        {"$group": {"_id": None, "avg": {"$avg": "$rating"}}}]))[0]["avg"]
    c1, c2 = st.columns(2)
    c1.metric("Average rating", f"{avg:.2f}")
    c2.metric("Reviews", f"{n:,}")

    se = list(db.drug_side_effects.find({"_id.drug": drug}).sort("value", -1).limit(8))
    if se:
        se_df = pd.DataFrame([{"side effect": s["_id"]["term"],
                               "% of this drug's reviews": round(s["value"] / n * 100, 1)}
                              for s in se])
        st.subheader("Most mentioned side effects")
        st.bar_chart(se_df.set_index("side effect"))
    else:
        st.info("No side-effect terms found for this drug.")

    st.subheader("Conditions this drug is reviewed for")
    st.dataframe(summary[summary["drug"] == drug]
                 .sort_values("review_count", ascending=False)
                 [["condition", "avg_rating", "review_count"]],
                 hide_index=True, use_container_width=True)