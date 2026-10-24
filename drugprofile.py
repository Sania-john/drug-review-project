from pymongo import MongoClient
from bson.code import Code
import matplotlib.pyplot as plt

db = MongoClient("mongodb://localhost:27017")["drug_db"]

# ---------------------------------------------------------------
# Job (c): side-effect term counts PER DRUG
# key = {drug, term}, value = number of reviews mentioning the term
# ---------------------------------------------------------------
map_c = Code("""function(){
  var terms = ["nausea","headache","dizziness","fatigue","insomnia","drowsiness","rash",
               "weight gain","dry mouth","anxiety","vomiting","diarrhea","constipation",
               "sweating","depression","pain","itching","cramps"];
  var text = this.review;
  var cond = this.condition;
  var drug = this.drugName;
  terms.forEach(function(t){
    if (text.indexOf(t) !== -1 && cond.indexOf(t) === -1) emit({drug: drug, term: t}, 1);
  });
}""")
reduce_c = Code("function(k, vals){ return Array.sum(vals); }")

db.command("mapReduce", "reviews", map=map_c, reduce=reduce_c, out="drug_side_effects")
print("Job (c) done: drug_side_effects collection created\n")

# ---------------------------------------------------------------
# Profile of the 8 most-reviewed drugs: avg rating + top 3 side effects
# ---------------------------------------------------------------
top_drugs = list(db.reviews.aggregate([
    {"$group": {"_id": "$drugName", "n": {"$sum": 1}, "avg": {"$avg": "$rating"}}},
    {"$sort": {"n": -1}}, {"$limit": 8}]))

profile = []
for d in top_drugs:
    se = list(db.drug_side_effects.find({"_id.drug": d["_id"]}).sort("value", -1).limit(3))
    text = ", ".join(f'{s["_id"]["term"]} ({s["value"] / d["n"] * 100:.0f}%)' for s in se)
    profile.append((d["_id"], d["n"], d["avg"], text))
    print(f'{d["_id"]:28} reviews={d["n"]:5}  avg={d["avg"]:.2f}  top side effects: {text}')

# Save the profile as its own MongoDB collection
db.drug_profile.drop()
db.drug_profile.insert_many([{"drug": p[0], "review_count": p[1],
                              "avg_rating": round(p[2], 2), "top_side_effects": p[3]}
                             for p in profile])

# Chart: avg rating of the most-reviewed drugs
plt.figure(figsize=(10, 5))
plt.barh([p[0] for p in profile], [p[2] for p in profile])
plt.gca().invert_yaxis()
plt.xlabel("Average rating")
plt.title("Most-reviewed drugs: average rating")
plt.tight_layout()
plt.savefig("drug_profiles.png")
plt.close()
print("\nSaved drug_profiles.png")