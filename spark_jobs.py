
import html
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_CSV = Path("drug review dataset.csv")
OUTPUT_DIR = Path("spark_output")

# Enable only if local MongoDB is running
WRITE_TO_MONGO = False
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "drug_db"

TERMS = [
    "nausea", "headache", "dizziness", "fatigue", "insomnia",
    "drowsiness", "rash", "weight gain", "dry mouth", "anxiety",
    "vomiting", "diarrhea", "constipation", "sweating",
    "depression", "pain", "itching", "cramps"
]


def find_terms(row):
    """Find configured side-effect terms in review text."""
    drug, condition, review, rating = row
    return [
        term for term in TERMS
        if term in review and term not in condition
    ]


def main():

    # -----------------------------------------------------
    # 1. CHECK INPUT DATASET
    # -----------------------------------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_CSV.resolve()}\n"
            "Keep 'drug review dataset.csv' in your project folder."
        )

    # -----------------------------------------------------
    # 2. START SPARK
    # -----------------------------------------------------
    spark = (
        SparkSession.builder
        .appName("DrugLensAI")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext

    try:
        # -------------------------------------------------
        # 3. LOAD AND CLEAN DATA
        # -------------------------------------------------
        df = pd.read_csv(INPUT_CSV)
        raw_count = len(df)

        required = [
            "condition", "drugName", "review", "rating", "uniqueID"
        ]

        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        df = df.dropna(
            subset=["condition", "drugName", "review", "rating"]
        ).copy()

        df = df[
            ~df["condition"].astype(str).str.contains(
                "users found this comment helpful",
                na=False
            )
        ].copy()

        df["review"] = df["review"].astype(str).apply(
            lambda value: html.unescape(value).replace('"', "").lower()
        )

        df["condition"] = (
            df["condition"].astype(str).str.strip().str.lower()
        )

        df["condition"] = df["condition"].replace({
            "mance anxiety": "performance anxiety"
        })

        df["drugName"] = (
            df["drugName"].astype(str).str.strip().str.lower()
        )

        df = df.drop_duplicates(subset="uniqueID").copy()

        df["rating"] = pd.to_numeric(
            df["rating"], errors="coerce"
        )

        df = df.dropna(subset=["rating"]).copy()

        print(f"Raw records: {raw_count}")
        print(f"Valid records after cleaning: {len(df)}")

        # RDD structure:
        # (drug, condition, review, rating)
        rows = list(zip(
            df["drugName"],
            df["condition"],
            df["review"],
            df["rating"]
        ))

        rdd = sc.parallelize(rows, 8).cache()

        # -------------------------------------------------
        # 4. JOB A: AVERAGE RATING PER DRUG AND CONDITION
        # Map -> ReduceByKey -> Average
        # -------------------------------------------------
        avg_rdd = (
            rdd.map(
                lambda row: (
                    (row[1], row[0]),
                    (float(row[3]), 1)
                )
            )
            .reduceByKey(
                lambda a, b: (a[0] + b[0], a[1] + b[1])
            )
            .mapValues(
                lambda value: (
                    round(value[0] / value[1], 2),
                    value[1]
                )
            )
            .cache()
        )

        print("\nJOB A: Drug rating summary")
        print("Condition-drug groups:", avg_rdd.count())

        # PYTHONHASHSEED must be set before Python starts.
        print(
            "Validation:",
            avg_rdd.lookup(("birth control", "etonogestrel"))
        )

        # -------------------------------------------------
        # 5. JOB B: SIDE-EFFECT COUNTS
        # -------------------------------------------------
        counts = (
            rdd.flatMap(
                lambda row: [
                    (term, 1) for term in find_terms(row)
                ]
            )
            .reduceByKey(lambda a, b: a + b)
            .cache()
        )

        print("\nJOB B: Top side-effect terms across all reviews")

        for term, count in counts.sortBy(
            lambda item: -item[1]
        ).take(10):
            print(f"{term:15} {count}")

        # -------------------------------------------------
        # 6. JOB B2: SIDE EFFECTS IN LOW-RATED REVIEWS
        # Rating <= 4
        # -------------------------------------------------
        low_counts = (
            rdd.filter(lambda row: float(row[3]) <= 4)
            .flatMap(
                lambda row: [
                    (term, 1) for term in find_terms(row)
                ]
            )
            .reduceByKey(lambda a, b: a + b)
            .cache()
        )

        print("\nTop side-effect terms for ratings <= 4")

        for term, count in low_counts.sortBy(
            lambda item: -item[1]
        ).take(10):
            print(f"{term:15} {count}")

        # -------------------------------------------------
        # 7. JOB C: SIDE-EFFECT COUNTS PER DRUG
        # -------------------------------------------------
        drug_se = (
            rdd.flatMap(
                lambda row: [
                    ((row[0], term), 1)
                    for term in find_terms(row)
                ]
            )
            .reduceByKey(lambda a, b: a + b)
            .cache()
        )

        print("\nJOB C: Drug-side-effect pairs")
        print("Total pairs:", drug_se.count())

        # -------------------------------------------------
        # 8. SAVE OUTPUT CSV FILES
        # -------------------------------------------------
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Average rating summary
        summary_rows = [
            {
                "condition": key[0],
                "drug": key[1],
                "avg_rating": value[0],
                "review_count": value[1]
            }
            for key, value in avg_rdd.collect()
        ]

        pd.DataFrame(
            summary_rows,
            columns=[
                "condition", "drug", "avg_rating", "review_count"
            ]
        ).to_csv(
            OUTPUT_DIR / "drug_summary.csv",
            index=False
        )

        # Overall side-effect counts
        pd.DataFrame(
            counts.collect(),
            columns=["term", "reviews"]
        ).sort_values(
            "reviews", ascending=False
        ).to_csv(
            OUTPUT_DIR / "side_effect_counts.csv",
            index=False
        )

        # Side-effect counts for low-rated reviews
        pd.DataFrame(
            low_counts.collect(),
            columns=["term", "reviews"]
        ).sort_values(
            "reviews", ascending=False
        ).to_csv(
            OUTPUT_DIR / "low_rating_side_effects.csv",
            index=False
        )

        # Side-effect counts per drug
        drug_rows = [
            {
                "drug": key[0],
                "term": key[1],
                "reviews": value
            }
            for key, value in drug_se.collect()
        ]

        pd.DataFrame(
            drug_rows,
            columns=["drug", "term", "reviews"]
        ).sort_values(
            "reviews", ascending=False
        ).to_csv(
            OUTPUT_DIR / "drug_side_effects.csv",
            index=False
        )

        print("\nAll Spark CSV outputs saved:")
        for file in sorted(OUTPUT_DIR.glob("*.csv")):
            print(f"{file.name}: {file.stat().st_size:,} bytes")

        # -------------------------------------------------
        # 9. OPTIONAL MONGODB EXPORT
        # -------------------------------------------------
        if WRITE_TO_MONGO:
            from pymongo import MongoClient

            with MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000
            ) as client:

                client.admin.command("ping")
                db = client[MONGO_DB]

                summary_docs = [
                    {
                        "condition": key[0],
                        "drug": key[1],
                        "avg_rating": value[0],
                        "review_count": value[1]
                    }
                    for key, value in avg_rdd.collect()
                ]

                side_effect_docs = [
                    {"term": term, "reviews": count}
                    for term, count in counts.collect()
                ]

                db.spark_drug_summary.delete_many({})
                if summary_docs:
                    db.spark_drug_summary.insert_many(summary_docs)

                db.spark_side_effects.delete_many({})
                if side_effect_docs:
                    db.spark_side_effects.insert_many(side_effect_docs)

                print("MongoDB export completed.")

        print("\nDrugLens AI Spark processing completed successfully!")

        rdd.unpersist()
        avg_rdd.unpersist()
        counts.unpersist()
        low_counts.unpersist()
        drug_se.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
