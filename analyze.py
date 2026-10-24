from pymongo import MongoClient
import matplotlib.pyplot as plt

db = MongoClient("mongodb://localhost:27017")["drug_db"]
MIN_REVIEWS = 10   # your dataset is ~53k rows, so 10 keeps enough conditions

# 1. Final collection {condition, drug, avg_rating, review_count}
final = [{"condition": d["_id"]["condition"], "drug": d["_id"]["drug"],
          "avg_rating": round(d["value"]["avg_rating"], 2),
          "review_count": int(d["value"]["review_count"])}
         for d in db.avg_rating_by_drug_condition.find()]
db.drug_summary.drop()
db.drug_summary.insert_many(final)
print("drug_summary documents:", db.drug_summary.count_documents({}))

# 2. Best drug per condition (top 10 most-reviewed conditions)
best = list(db.drug_summary.aggregate([
    {"$match": {"review_count": {"$gte": MIN_REVIEWS}}},
    {"$sort": {"condition": 1, "avg_rating": -1}},
    {"$group": {"_id": "$condition", "best_drug": {"$first": "$drug"},
                "avg_rating": {"$first": "$avg_rating"},
                "reviews": {"$first": "$review_count"}}},
    {"$sort": {"reviews": -1}}, {"$limit": 10}]))
print("\nBest drug per condition:")
for b in best:
    print(b)

# 3. Validation: compare MapReduce vs direct $avg for the most-reviewed pair
top = db.drug_summary.find().sort("review_count", -1).limit(1)[0]
direct = list(db.reviews.aggregate([
    {"$match": {"drugName": top["drug"], "condition": top["condition"]}},
    {"$group": {"_id": None, "avg": {"$avg": "$rating"}, "n": {"$sum": 1}}}]))[0]
print("\nValidation for", top["drug"], "/", top["condition"])
print("MapReduce :", top["avg_rating"], top["review_count"])
print("Direct avg:", round(direct["avg"], 2), direct["n"])
print("MATCH" if round(direct["avg"], 2) == top["avg_rating"] and direct["n"] == top["review_count"] else "MISMATCH")

# 4. Chart 1: top side-effect terms
se = list(db.side_effect_counts.find().sort("value", -1).limit(10))
plt.figure(figsize=(9, 5))
plt.bar([s["_id"] for s in se], [s["value"] for s in se])
plt.title("Most mentioned side-effect terms")
plt.ylabel("Number of reviews"); plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig("side_effects.png"); plt.close()

# 5. Chart 2: best drug per condition
plt.figure(figsize=(10, 5))
plt.barh([f'{b["_id"]} ({b["best_drug"]})' for b in best], [b["avg_rating"] for b in best])
plt.title("Highest-rated drug for the 10 most-reviewed conditions")
plt.xlabel("Average rating"); plt.gca().invert_yaxis()
plt.tight_layout(); plt.savefig("best_drugs.png"); plt.close()