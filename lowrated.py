from pymongo import MongoClient
from bson.code import Code
import matplotlib.pyplot as plt

db = MongoClient("mongodb://localhost:27017")["drug_db"]

# Same side-effect job as mapreduce.py (job b), run only on low-rated reviews
map_b = Code("""function(){
  var terms = ["nausea","headache","dizziness","fatigue","insomnia","drowsiness","rash",
               "weight gain","dry mouth","anxiety","vomiting","diarrhea","constipation",
               "sweating","depression","pain","itching","cramps"];
  var text = this.review;
  var cond = this.condition;
  terms.forEach(function(t){
    if (text.indexOf(t) !== -1 && cond.indexOf(t) === -1) emit(t, 1);
  });
}""")

reduce_b = Code("function(k, vals){ return Array.sum(vals); }")

db.command("mapReduce", "reviews",
           map=map_b, reduce=reduce_b,
           query={"rating": {"$lte": 4}},
           out="side_effect_low_rated")
print("Low-rated MapReduce done")

# Percentage of reviews mentioning each term
low_total = db.reviews.count_documents({"rating": {"$lte": 4}})
all_total = db.reviews.count_documents({})
print("Low-rated reviews:", low_total, "| All reviews:", all_total, "\n")

low = {d["_id"]: d["value"] / low_total * 100 for d in db.side_effect_low_rated.find()}
allr = {d["_id"]: d["value"] / all_total * 100 for d in db.side_effect_counts.find()}

terms = sorted(low, key=low.get, reverse=True)[:10]
print(f"{'term':15} {'low-rated %':>12} {'all reviews %':>14}")
for t in terms:
    print(f"{t:15} {low[t]:11.1f}% {allr.get(t, 0):13.1f}%")

# Chart: low-rated vs all reviews
x = range(len(terms))
plt.figure(figsize=(10, 5))
plt.bar([i - 0.2 for i in x], [low[t] for t in terms], width=0.4, label="Low-rated (<=4)")
plt.bar([i + 0.2 for i in x], [allr.get(t, 0) for t in terms], width=0.4, label="All reviews")
plt.xticks(list(x), terms, rotation=45)
plt.ylabel("% of reviews mentioning term")
plt.title("Side-effect terms: low-rated vs all reviews")
plt.legend()
plt.tight_layout()
plt.savefig("side_effects_low_vs_all.png")
plt.close()
print("\nSaved side_effects_low_vs_all.png")