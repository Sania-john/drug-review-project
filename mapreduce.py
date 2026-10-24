from pymongo import MongoClient
from bson.code import Code

db = MongoClient("mongodb://localhost:27017")["drug_db"]
print("MongoDB version:", db.client.server_info()["version"])

# ---------------------------------------------------------------
# Job (a): average rating per drug and condition
# ---------------------------------------------------------------
map_a = Code("""function(){
  emit({condition: this.condition, drug: this.drugName},
       {sum: this.rating, count: 1});
}""")

reduce_a = Code("""function(key, vals){
  var r = {sum: 0, count: 0};
  vals.forEach(function(v){ r.sum += v.sum; r.count += v.count; });
  return r;
}""")

finalize_a = Code("""function(key, v){
  return {avg_rating: v.sum / v.count, review_count: v.count};
}""")

db.command("mapReduce", "reviews",
           map=map_a, reduce=reduce_a, finalize=finalize_a,
           out="avg_rating_by_drug_condition")
print("Job (a) done")

# ---------------------------------------------------------------
# Job (b): side-effect term counts
# A term is NOT counted if it is part of the drug's own condition
# (e.g. "depression" in a review of an antidepressant for depression)
# ---------------------------------------------------------------
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
           out="side_effect_counts")
print("Job (b) done")