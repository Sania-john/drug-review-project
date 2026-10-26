import os
import pandas as pd
from pymongo import MongoClient

db = MongoClient("mongodb://localhost:27017")["drug_db"]
os.makedirs("powerbi_data", exist_ok=True)

# 1. Best-drug table: condition, drug, avg_rating, review_count
summary = pd.DataFrame(list(db.drug_summary.find({}, {"_id": 0})))
summary.to_csv("powerbi_data/drug_summary.csv", index=False)

# 2. Side effects: all reviews vs low-rated reviews (in % of reviews)
all_total = db.reviews.count_documents({})
low_total = db.reviews.count_documents({"rating": {"$lte": 4}})
allr = {d["_id"]: d["value"] / all_total * 100 for d in db.side_effect_counts.find()}
low = {d["_id"]: d["value"] / low_total * 100 for d in db.side_effect_low_rated.find()}
compare = pd.DataFrame([{"term": t,
                         "all_reviews_pct": round(allr.get(t, 0), 2),
                         "low_rated_pct": round(low.get(t, 0), 2)} for t in allr])
compare.to_csv("powerbi_data/side_effects_compare.csv", index=False)

# 3. Side effects per drug (flatten the {drug, term} key)
drug_se = pd.DataFrame([{"drug": d["_id"]["drug"], "term": d["_id"]["term"],
                         "mentions": int(d["value"])}
                        for d in db.drug_side_effects.find()])
drug_se.to_csv("powerbi_data/drug_side_effects.csv", index=False)

print("Exported to powerbi_data/:")
print("  drug_summary.csv          ", len(summary), "rows")
print("  side_effects_compare.csv  ", len(compare), "rows")
print("  drug_side_effects.csv     ", len(drug_se), "rows")