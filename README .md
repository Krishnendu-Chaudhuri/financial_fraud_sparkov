# Complete Financial Fraud Detection Analysis

Classical ML · Deep Learning · Graph Neural Networks

End-to-end notebook that detects rare fraudulent credit-card transactions (`is_fraud = 1`) on the **Sparkov** dataset and compares four complementary modelling families under rare-event metrics.

| Model family | Technique | Why it helps for fraud |
|--------------|-----------|------------------------|
| Classical | Logistic Regression, Random Forest, **XGBoost** | Strong tabular baselines; fast; interpretable feature importance |
| Deep Learning | PyTorch **Embedding MLP** | Learns dense representations of high-cardinality categoricals (merchant, city, job, …) |
| Graph Neural Network | Bipartite **GraphSAGE** (card ↔ merchant) | Captures fraud rings / shared-merchant patterns that pure tabular models miss |
| Hybrid | GNN node embeddings → XGBoost | Combines relational structure with powerful tree boosting |

---

## Dataset

- **Source**: Sparkov credit-card transactions 
- **Primary file used**: `fraudTrain.csv`
- **Size (as run in the notebook)**: ~1.3M rows × 23 columns
- **Fraud rate**: ~0.58% (strong class imbalance)

Typical columns include:

`trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `gender`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `unix_time`, `merch_lat`, `merch_long`, `is_fraud`

---

## Notebook structure

| Section | Content |
|---------|---------|
| **0** | Environment & imports (reproducibility seeds, device detection) |
| **1** | Load Sparkov data + basic sanity checks |
| **2** | Exploratory Data Analysis (class imbalance, amount distributions, category risk, etc.) |
| **3–5** | Feature engineering & train/validation/test split (leakage-safe) |
| **6** | Classical models – Logistic Regression, Random Forest, XGBoost |
| **7** | Deep Learning – Embedding MLP (PyTorch) |
| **8** | Graph Neural Network – bipartite GraphSAGE (card–merchant) |
| **9** | Hybrid – GNN embeddings fed into XGBoost |
| **10** | Fair comparison – ROC / PR curves, PR-AUC, threshold tuning |
| **11** | Key take-aways |

---

## Requirements

```text
python >= 3.10
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
torch
# Optional but recommended for full GNN path:
torch-geometric   # or a pure-PyTorch GraphSAGE implementation used in the notebook
```

Install example:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn torch
# If using torch-geometric:
pip install torch-geometric
```

---

## How to run

1. Download the Sparkov dataset from Kaggle and place `fraudTrain.csv` somewhere accessible.
2. Open `02_complete_fraud_analysis.ipynb`.
3. Update the data path in the load cell (Section 1):

   ```python
   DATA_DIR = Path.home() / "Downloads" / "Sparkov data"
   DATA_PATH = DATA_DIR / "fraudTrain.csv"
   ```

4. Run all cells top-to-bottom.  
   - Training is CPU-friendly; a GPU speeds up the MLP / GNN stages if available.
   - The notebook seeds NumPy and PyTorch (`RANDOM_STATE = 42`) for reproducibility.

---

## Evaluation protocol (important)

Because fraud is rare (~0.6%):

- **Primary metrics**: PR-AUC (Average Precision) and ROC-AUC  
- Thresholds are chosen on a **validation** set (not the test set)  
- Class weights / scale_pos_weight are used where appropriate  
- Encoders, scalers, and the bipartite graph adjacency are fit **only on training data** to avoid leakage

Side-by-side ROC and Precision-Recall curves for all six models are produced at the end of the notebook.

---

## Key take-aways (from the notebook)

1. Class imbalance dominates – always report **PR-AUC** and a validation-chosen threshold, not accuracy.
2. **XGBoost** remains an extremely strong and practical baseline for tabular fraud data.
3. **Embedding MLP** learns useful representations of high-cardinality categoricals and often matches or exceeds tree models when enough data is present.
4. **GraphSAGE** on the card–merchant bipartite graph can surface relational fraud patterns (shared compromised merchants, fraud rings) that pure tabular models miss.
5. **Hybrid** (GNN embeddings + XGBoost) is a simple way to combine the best of both worlds and is frequently used in production pipelines.
6. **Leakage control** is non-negotiable: fit encoders/scalers and build the graph adjacency only on training data.

---

## File layout

```text
.
├── 02_complete_fraud_analysis.ipynb   # Main analysis notebook
└── README.md                          # This file
```

---

## Notes & caveats

- The notebook was developed against the full Sparkov training split (~1.3M rows). Sub-sampling is possible for faster iteration but will change absolute metric values.
- Graph construction uses a bipartite card ↔ merchant graph; edge features / temporal edges can be added as future work.
- Production deployment would additionally require online feature stores, concept-drift monitoring, and human-in-the-loop review queues – topics outside the scope of this analysis notebook.

---
