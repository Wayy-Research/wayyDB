For anyone who wants to reproduce the benchmarks:

```
pip install wayy-db[bench]
python -m benchmarks.benchmark --compare pandas,polars,duckdb
```

Takes about 2 minutes. I'd love to see results on different hardware!

You can also run individual benchmarks:
```
python -m benchmarks.benchmark --only asof_join
python -m benchmarks.benchmark --quick  # faster with smaller datasets
```
