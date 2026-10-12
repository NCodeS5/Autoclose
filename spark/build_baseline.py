"""PySpark batch job: builds the per-vendor baseline from a (large) historical ledger CSV.

    python spark/build_baseline.py --demo                       # generates a synthetic history, then runs
    python spark/build_baseline.py --input ledger_history.csv   # your own CSV: vendor,category,amount

Output: vendor_baseline.csv (vendor, mean, sd, category). Load it with Baseline.from_csv(...).
Runs in local mode here; the same code runs unchanged on a cluster."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspark.sql import SparkSession, functions as F, Window

def build(spark, input_csv, output_csv):
    df = spark.read.csv(input_csv, header=True, inferSchema=True)
    stats = df.groupBy("vendor").agg(F.avg("amount").alias("mean"), F.stddev_pop("amount").alias("sd"))
    counts = df.groupBy("vendor", "category").count()
    w = Window.partitionBy("vendor").orderBy(F.desc("count"), F.asc("category"))
    mode = counts.withColumn("rn", F.row_number().over(w)).filter("rn = 1").select("vendor", "category")
    out = stats.join(mode, "vendor").orderBy("vendor")
    out.toPandas().to_csv(output_csv, index=False)
    return out.count()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ledger_history.csv")
    ap.add_argument("--output", default="vendor_baseline.csv")
    ap.add_argument("--demo", action="store_true", help="generate a synthetic history first")
    a = ap.parse_args()
    if a.demo:
        import pandas as pd
        from autoclose.generate_data import generate
        rows = [dict(vendor=e["vendor"], category=e["category"], amount=e["amount"])
                for r in generate(60, anomaly_rate=0.0, seed=99) for e in r.ledger]
        pd.DataFrame(rows).to_csv(a.input, index=False)
        print(f"wrote {len(rows)} ledger rows to {a.input}")
    spark = SparkSession.builder.master("local[*]").appName("autoclose-baseline").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    n = build(spark, a.input, a.output)
    print(f"wrote baseline for {n} vendors to {a.output}")
    spark.stop()
