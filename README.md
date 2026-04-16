# MVTec Anomaly Detection (Nested Learning / CMS)

Minimal instructions to run the MVTec pipeline.

Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Quick verify

```bash
python dataset/download_mvtec.py           # one-time
python scripts/01_data_preparation.py --run_verify --config ./conf/config.yaml
```

Run training/eval

```bash
python training/trainer.py --config ./conf/config.yaml
python training/evaluator.py --config ./conf/config.yaml
```

Notes

- Config defaults live in `conf/config.yaml`.

Project layout

```
conf/  dataset/  models/  scripts/  training/  results/
```
