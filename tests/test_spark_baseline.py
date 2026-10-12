import pytest
pytest.importorskip("pyspark")
import pandas as pd
from autoclose.generate_data import generate
from autoclose.tools import Baseline

def test_spark_baseline_matches_python_baseline(tmp_path):
    from pyspark.sql import SparkSession
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "build_baseline", pathlib.Path(__file__).parent.parent / "spark" / "build_baseline.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    warm = generate(60, anomaly_rate=0.0, seed=99)
    src, out = tmp_path / "hist.csv", tmp_path / "base.csv"
    pd.DataFrame([dict(vendor=e["vendor"], category=e["category"], amount=e["amount"])
                  for r in warm for e in r.ledger]).to_csv(src, index=False)
    spark = SparkSession.builder.master("local[1]").appName("t").getOrCreate()
    try:
        assert mod.build(spark, str(src), str(out)) == 10
    finally:
        spark.stop()
    a, b = Baseline(warm).stats, Baseline.from_csv(str(out)).stats
    assert a.keys() == b.keys()
    for v in a:
        assert a[v][0] == pytest.approx(b[v][0], rel=1e-6)
        assert a[v][1] == pytest.approx(b[v][1], rel=1e-6)
        assert a[v][2] == b[v][2]
