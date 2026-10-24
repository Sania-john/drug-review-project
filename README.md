# Drug Review Effectiveness and Side-Effect Analyzer

Big data project using **MongoDB** and **MapReduce** on the UCI ML Drug Review dataset (Drugs.com reviews).

## Dataset
- Source: [PASTE KAGGLE LINK HERE]
- File used: `drug review dataset.csv` (53,766 rows, 53,200 after cleaning)
- Columns: `uniqueID, drugName, condition, review, rating, date, usefulCount`
- The CSV is not included in this repo. Download it from Kaggle and place it in the project folder.

## Tools
Python 3, PyMongo, pandas, matplotlib, MongoDB 8.2.0 (mapReduce command), VS Code

## Project files
| File | Purpose |
|---|---|
| `load.py` | Cleans the CSV and loads it into MongoDB (`drug_db.reviews`) |
| `mapreduce.py` | Job (a): average rating per drug and condition. Job (b): side-effect term counts |
| `analyze.py` | Builds `drug_summary` `{condition, drug, avg_rating, review_count}`, ranks the best drug per condition, validates MapReduce against `$avg`, draws charts |
| `lowrated.py` | Re-runs the side-effect job on reviews rated 4 or below and compares with all reviews |
| `drugprofile.py` | Job (c): side-effect counts per drug, saved to `drug_profile` |

## How to run
```bash
pip install pymongo pandas matplotlib
python load.py
python mapreduce.py
python analyze.py
python lowrated.py
python drugprofile.py
```
MongoDB must be running on `mongodb://localhost:27017`.

## Method notes
- Side-effect terms are matched by keyword. A term is not counted when it appears in the drug's own condition (for example, "depression" for a depression drug).
- Best drug per condition uses a minimum of 10 reviews.
- Validation: the MapReduce average and review count match a direct `$avg` aggregation (etonogestrel / birth control: 5.89 over 1,080 reviews).

## Key results
- Most mentioned side-effect terms: pain, headache, nausea, anxiety, cramps.
- Itching, diarrhea and headache are the most over-represented in low-rated reviews (about 1.3 to 1.5 times their overall rate).
- Highest-rated drug for performance anxiety: propranolol (9.67 average).

## Limitations
- Keyword matching misses misspellings and negation ("no nausea").
- Ratings measure patient satisfaction, not clinical effectiveness. This is not medical advice.
- Brand and generic names (for example, nexplanon and etonogestrel) are counted separately.
- MongoDB has deprecated `mapReduce` since 5.0, though it still works in 8.2.

## Author
[YOUR NAME, ID]