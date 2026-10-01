<div align="center">

# 📊 Customer Churn Intelligence & Management System

### سامانه تحلیل هوشمند و مدیریت ریزش مشتری

**Monitor customer behaviour · Predict churn risk · Recommend a win-back action — in one dashboard.**


[Executive Summary](#-executive-summary--the-business-problem) ·
[Architecture](#%EF%B8%8F-system-architecture--pipeline) ·
[Features](#-key-dashboard-features) ·
[Quickstart](#-quickstart-guide) ·
[Structure](#%EF%B8%8F-project-structure) ·
[Disclaimer](#-privacy-statement--disclaimer)

</div>



---

## 🎯 Executive Summary & The Business Problem

Most online retailers and small merchants only discover that a customer has left **after** the last order was placed. By then the revenue is gone and winning the customer back is expensive.

This project turns historical behaviour into an **early-warning system**. It scores each customer's probability of leaving, explains the main reason, and drafts a tailored win-back offer — so a merchant can act while the customer is still reachable.

### What the data says (5,630 customers)

| Finding | Evidence |
|---|---|
| 📉 Baseline churn | **16.84 %** of customers left (948 of 5,630) |
| ⚠️ Complaints are the strongest warning sign | **31.7 %** churn with a complaint vs **10.9 %** without (≈ **2.9×**); 53.6 % of all churners had complained |
| 🧭 The first month is critical | **51.8 %** churn for customers with ≤ 1 month of tenure; **65.5 %** of all churners were in their first month |
| 😀 The satisfaction paradox | Customers rating **5/5** churn at **23.8 %** vs **11.5 %** for those rating **1/5** — surveys alone are misleading |
| 🛍️ Category risk differs sharply | Mobile & digital **27.4 %**, fashion 15.5 %, laptops 10.2 %, grocery **4.9 %** |
| 🚚 Logistics and rewards matter | Churn rises from 13.5 % to 20.4 % as warehouse distance grows; it falls from 29.5 % to 10.2 % as cashback rises |

> [!IMPORTANT]
> These are statistical associations in one dataset. They show *where* risk concentrates, not proven causes.


---

## 🏗️ System Architecture & Pipeline

```text
 ┌──────────────────────────── OFFLINE (run once) ────────────────────────────┐
 │                                                                            │
 │  data/E-Commerce-Dataset.csv          5,630 rows × 20 columns              │
 │            │                                                               │
 │            ▼                                                               │
 │  export_model.py ── fit_preprocessing()                                    │
 │     1. drop CustomerID                                                     │
 │     2. median imputation (numeric)                                         │
 │     3. merge duplicate labels  (Mobile→Mobile Phone, CC→Credit Card, …)    │
 │     4. IQR outlier capping on 9 continuous columns                         │
 │     5. feature engineering: avg_cashbk_per_order = Cashback / OrderCount   │
 │     6. one-hot encoding (drop_first=True)  →  27 features                  │
 │            │                                                               │
 │            ▼                                                               │
 │     stratified 80 / 20 split (random_state = 42)                           │
 │     RandomForestClassifier(n_estimators=300, class_weight="balanced")      │
 │            │                                                               │
 │            ├──► models/churn_model.joblib   (model + feature list +        │
 │            │                                 medians + IQR bounds +        │
 │            │                                 category levels + metadata)   │
 │            └──► data/test_stream_data.csv   (200 held-out raw rows)        │
 └────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────── ONLINE (runtime) ──────────────────────────────┐
 │                                                                            │
 │  raw customer record (dict)                                                │
 │            │                                                               │
 │            ▼                                                               │
 │  churn_inference.predict_single_customer()                                 │
 │     • re-applies the SAME preprocessing from the saved bundle              │
 │     • predicts churn probability                                           │
 │     • tree-path attribution → Top-3 risk drivers                           │
 │            │                                                               │
 │            ▼                                                               │
 │  app.py  (Streamlit + Plotly)                                              │
 │     ├─ Tab 1  Strategic Intelligence  (KPIs · 5 analyses · roadmap)        │
 │     └─ Tab 2  Live Radar  +  Retention Lab                                 │
 │                  radar: 2 customers / 6 s from test_stream_data.csv        │
 │                  lab : 7-factor form → risk · cause · value · SMS draft    │
 └────────────────────────────────────────────────────────────────────────────┘
```

**Design principles**

- **One preprocessing path.** Medians, IQR bounds and category levels are fitted once and stored *inside* the model bundle, so training and inference cannot drift apart.
- **Raw in, explained out.** The engine always receives raw column values; Persian labels, payment-method names and Toman conversion are applied only in the display layer.
- **Explainability without extra libraries.** Risk drivers come from a tree-path (Saabas-style) attribution implemented in plain NumPy; contributions sum exactly to the predicted probability.

### Model snapshot

| Item | Value |
|---|---|
| Algorithm | Random Forest Classifier, 300 trees, `class_weight="balanced"` |
| Split | Stratified 80 / 20 → 4,504 train · 1,126 test |
| Test accuracy | **≈ 98 %** |
| ROC-AUC | **≈ 0.998** |
| Churn class (test) | precision ≈ 1.00 · recall ≈ 0.88 · F1 ≈ 0.94 |
| Input features | 27 after encoding (18 raw + 1 engineered, one-hot expanded) |

Reproduce the numbers with `python export_model.py` (it prints the full evaluation report).

---

## ✨ Key Dashboard Features

### 📊 Tab 1 — Strategic Intelligence (executive view)

| Component | What it delivers |
|---|---|
| **KPI strip** | Overall churn rate, number of churned customers, annual value at risk (adjustable assumption), pattern-detection strength, prediction accuracy |
| **5 deep analyses** | Each with numbers, takeaways and an interactive Plotly chart (see below) |
| **4-step roadmap** | Concrete actions for Marketing, Support, Logistics and Loyalty teams |

**The five analyses**

1. 😀 **Satisfaction paradox** — highest-rated customers churn about twice as often as the lowest-rated ones ("silent churn").
2. ⚠️ **Complaint lever** — one complaint multiplies churn risk by ≈ 3; response time is the cheapest retention lever.
3. 🧭 **First-month cliff** — risk is concentrated in the first 30 days of membership.
4. 🛍️ **Category vulnerability** — mobile & digital are the riskiest, grocery the most loyal.
5. 🚚 **Logistics & rewards** — warehouse distance and cashback level both move churn.

### 🎯 Tab 2 — Operational Radar & Retention Lab

**Live Dual Radar**

- Always monitors **exactly two customers**, replaced by two new ones from the stream every **6 seconds**.
- Animated **countdown bar** shows time to the next scan; a toggle pauses/resumes auto-refresh.
- Minimal cards with a colour-coded status:

| Status | Risk range | Card shows |
|---|---|---|
| 🔴 Critical | > 70 % | percentage + progress bar |
| 🟠 Warning | 35 – 70 % | percentage + progress bar |
| 🟢 Stable | < 35 % | **no percentage** — "stable" label only |

- **"View full information"** expands all 19 real fields of the customer (membership, city tier, warehouse distance, payment method, satisfaction, complaints, cashback in Toman, …). While details are open the radar freezes so they stay readable.

**Interactive Retention Lab** (static `st.form` — nothing is computed until you press *Calculate*)

- **7 key factors:** tenure (months) · recent complaint · product category · days since last purchase · city tier · satisfaction score (1–5) · total previous orders (+ business name for the message).
- Remaining fields are auto-filled with **medians of similar historical customers**.
- **Output panel:** churn probability and status · root-cause analysis · **estimated retention value** (illustrative) · personalised SMS draft · *Register win-back scenario* button.
- **Tiered offers:**

| Risk level | Offer | Code |
|---|---|---|
| 🔴 Critical | 10 % goodwill discount + follow-up wording | `VIP-CARE10` (gold) |
| 🟠 Warning | 5 % reminder discount | `RETURN5` (silver) |
| 🟢 Stable | No action needed | — |

- Three **quick-scenario buttons** (complaining buyer · stable customer · inactive user) pre-fill the form for live demos.

---

## 🚀 Quickstart Guide

**Prerequisites:** Python 3.10 or newer (developed on 3.12) and Git.

### 1 · Clone the repository

```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
```

### 2 · Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3 · Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>Expected contents of <code>requirements.txt</code></summary>

```text
streamlit>=1.40
plotly
pandas
numpy
scikit-learn
joblib
```

</details>

### 4 · (Optional) Rebuild the model and the demo stream

The repository already contains `models/churn_model.joblib` and `data/test_stream_data.csv`. Rebuild them if you change the data or your scikit-learn version differs from the one used to train the model (pickled models are version-sensitive):

```bash
python export_model.py                       # uses data/E-Commerce-Dataset.csv
python export_model.py --data path/to/your.csv
```

### 5 · Smoke-test the prediction engine

```bash
python test_run.py
```

It scores one sample customer and prints the churn probability and the top risk drivers.

### 6 · Launch the dashboard

```bash
streamlit run app.py
```


### Use the engine programmatically

```python
from churn_inference import predict_single_customer

result = predict_single_customer(customer_dict)   # raw dataset columns
# {
#   "churn_probability_pct": 63.33,
#   "top_risk_drivers": [{"feature": "Complain", "value": 1, "impact_pct_points": 9.41}, ...]
# }
```

---

## 🗂️ Project Structure

```text
.
├── app.py                      # Streamlit dashboard (both tabs, display layer)
├── churn_inference.py          # predict_single_customer(): preprocessing + prediction + Top-3 drivers
├── export_model.py             # offline training/export: model bundle + demo stream
├── test_run.py                 # quick smoke test of the inference engine
├── requirements.txt            # Python dependencies
├── data/
│   ├── E-Commerce-Dataset.csv  # raw training data (5,630 customers)
│   └── test_stream_data.csv    # 200 held-out raw rows used as the live stream
├── models/
│   └── churn_model.joblib      # trained model + preprocessing parameters + metadata
└── notebooks/
    └── churn_porject.ipynb     # exploratory analysis and model development
```

| Path | Role |
|---|---|
| `app.py` | Presentation only: charts, cards, radar, lab. Sends **raw** values to the engine. |
| `churn_inference.py` | Single source of truth for inference. Loads the bundle once (cached). |
| `export_model.py` | Reproducible pipeline: fit preprocessing → train → evaluate → export. |
| `data/` | Training data and the simulated stream. |
| `models/` | Versioned model artefact. |
| `notebooks/` | Research notebook behind the five analyses. |


---

## 🔒 Privacy Statement & Disclaimer

**Data**

- The dataset contains anonymous numeric customer IDs and behavioural attributes only — no names, phone numbers, e-mail addresses or other direct identifiers. It originates from a third-party e-commerce churn dataset; check its license before redistributing it.


