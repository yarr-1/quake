## Quake Forecast System

this repository contains the code for:

* **autoencoder anomaly detection** for EM emissions, atmospheric pressure, and TEC
* **decision tree** to classify if a measurement is anomalous or regular
* **LSTM / probabilistic forecasting**

---

### Repository structure

| folder       | purpose                                      |
| ------------ | -------------------------------------------- |
| `data/`      | raw CSVs (input)                             |
| `models/`    | saved models: autoencoder, scaler, DT        |
| `pipeline/`  | Python scripts (train / test / labeling etc) |
| `notebooks/` | experiments, prototyping                     |

---

### Step 1 — Train Autoencoder

```bash
python pipeline/train_autoencoder.py
```

this outputs:

* `models/autoencoder.keras`
* `models/scaler.save`

---

### Step 2 — Generate anomaly labels

```bash
python pipeline/make_labels.py
```

this outputs:

* decision_tree_input.csv

---

### Step 3 — Train Decision Trees

```bash
python pipeline/train_decision_trees.py
```

This outputs:

* `models/`decision_tree_model.pkl

---

next:

* push new code changes into this repo
* build the frontend (Vercel)
* API endpoint that loads dt.pkl + parses last 24 hours + returns anomaly flags
