import pandas as pd, html
from pymongo import MongoClient

df = pd.read_csv("drug review dataset.csv")
raw = len(df)

df = df.dropna(subset=["condition", "drugName", "review", "rating"])
df = df[~df["condition"].str.contains("users found this comment helpful", na=False)]
df["review"] = df["review"].apply(lambda t: html.unescape(t).replace('"', '').lower())
df["condition"] = df["condition"].str.strip().str.lower()
df["condition"] = df["condition"].replace({"mance anxiety": "performance anxiety"})
df["drugName"] = df["drugName"].str.strip().str.lower()
df = df.drop_duplicates(subset="uniqueID")
df["rating"] = df["rating"].astype(float)
print(raw, "->", len(df), "records after cleaning")

db = MongoClient("mongodb://localhost:27017")["drug_db"]
db.reviews.drop()
db.reviews.insert_many(df.to_dict("records"))
db.reviews.create_index([("condition", 1), ("drugName", 1)])
print("In MongoDB:", db.reviews.count_documents({}))