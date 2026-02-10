# Model Evaluation Criteria

*Purpose: Define success metrics before building*

---

## 🎯 Project Goal

Build a graph-based fraud detection model for Medicare Part B skin substitute billing that:
1. Demonstrates technical capability
2. Produces results worth presenting at conferences
3. Potentially generates actionable intelligence for investigators

---

## 📊 Available Labels

### Primary: LEIE Exclusions
- **Source:** HHS-OIG List of Excluded Individuals/Entities
- **URL:** https://oig.hhs.gov/exclusions/exclusions_list.asp
- **What it contains:** NPIs of providers excluded from federal healthcare programs
- **Match method:** Join on NPI to Medicare billing data

### Label Challenges

| Challenge | Impact |
|-----------|--------|
| **Incomplete** | Not all fraud is caught; many fraudsters never excluded |
| **Lagged** | Exclusions happen 2-5 years after fraud occurred |
| **Sparse** | Estimated 0.1-0.5% of providers are labeled fraudsters |
| **Selection bias** | Only caught fraud is labeled; sophisticated fraud escapes |

---

## 📈 Evaluation Metrics

### 1. AUC-ROC (Area Under ROC Curve)
**Question:** Can the model rank fraudsters higher than non-fraudsters?

| Score | Interpretation |
|-------|----------------|
| 0.50 | Random (no signal) |
| 0.60 | Weak signal |
| 0.70 | Moderate signal |
| 0.80 | Strong signal |
| 0.90+ | Excellent |

**Target:** AUC > 0.75

### 2. Precision@K
**Question:** If we flag top K providers, what % are actually fraudulent?

**Target:** Precision@100 > 5% (10x lift over random)

### 3. Lift
**Question:** How much better than random are we?

**Target:** Lift@100 > 10x

---

## 🔬 Experimental Design

### Experiment 1: Supervised Classification

**Models to compare:**
1. Random baseline
2. Heuristic baseline (flag highest billers)
3. XGBoost (tabular) — provider features only
4. GraphSAGE (GNN) — with network structure
5. GAT (GNN) — with attention

**Key question:** Does GNN beat XGBoost?

### Experiment 2: Anomaly Detection (Unsupervised)

**Setup:**
- Graph autoencoder
- High reconstruction error = anomaly
- No labels needed

### Experiment 3: Known Case Validation

**Question:** Can we flag known DOJ cases?

---

## ✅ Success Criteria

| Level | Criteria |
|-------|----------|
| **Minimum** | GNN AUC > 0.65, beats XGBoost |
| **Good** | AUC > 0.75, Precision@100 > 5% |
| **Excellent** | AUC > 0.85, flags known DOJ cases |

---

## 🧠 Key Insight

> The goal is proving that **graph structure adds value** over tabular methods. If GNN beats XGBoost, we've demonstrated that network analysis catches fraud patterns that individual provider analysis misses.
