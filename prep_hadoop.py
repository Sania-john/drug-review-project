import csv
import html
import pandas as pd

# Cleans the dataset (same steps as load.py) and writes a tab-separated file
# for Hadoop: drugName <TAB> condition <TAB> rating <TAB> review
df = pd.read_csv("drug review dataset.csv")
raw = len(df)

df = df.dropna(subset=["condition", "drugName", "review", "rating"])
df = df[~df["condition"].str.contains("users found this comment helpful", na=False)]
df = df.drop_duplicates(subset="uniqueID")


def one_line(text):
    # lower case, decode HTML, remove quotes, collapse tabs/newlines into single spaces
    return " ".join(html.unescape(str(text)).replace('"', "").lower().split())


df["review"] = df["review"].apply(one_line)
df["condition"] = df["condition"].apply(one_line)
df["drugName"] = df["drugName"].apply(one_line)
df["condition"] = df["condition"].replace({"mance anxiety": "performance anxiety"})
df["rating"] = df["rating"].astype(float)

df[["drugName", "condition", "rating", "review"]].to_csv(
    "reviews_clean.tsv", sep="\t", index=False, header=False, quoting=csv.QUOTE_NONE)

print(raw, "->", len(df), "records written to reviews_clean.tsv")
