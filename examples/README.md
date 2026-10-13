# Synthetic examples — software demonstration only

Every CSV and model in this directory is generated from the synthetic traffic model in `ids/demo.py`. No public benchmark dataset or campus traffic was used. The associated metrics are not research evidence.

`canonical_train.csv`, `canonical_calibration.csv`, `canonical_validation.csv`, and `canonical_test.csv` contain 1,200 / 400 / 400 / 400 independently generated rows respectively. Train and evaluate this example with:

```sh
python -m ids.cli train --train examples/canonical_train.csv --calibration examples/canonical_calibration.csv --validation examples/canonical_validation.csv --test examples/canonical_test.csv --schema canonical-pcap-v1 --out models --budget 0.15
```

The included `synthetic_run/` preserves example results and the three trained example models. They were produced with the versions listed in `VALIDATION.md` and may require retraining with a different environment. For a quick dashboard test, create a `models` folder, copy `synthetic_run/selected.joblib` there, select that model, and upload `canonical_test.csv`. The bundled weights carry demo=True, so the dashboard and exports retain the synthetic-source warning; these weights are not suitable for operational traffic.

Import `synthetic_predictions.csv` into the hosted dashboard to inspect **real model outputs from the synthetic example**, including reference-replacement explanations for 50 rows. The hosted importer reads the demo flag and retains the synthetic demonstration banner. Do not confuse this with actual UNSW or CIC results.
