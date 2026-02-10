# Skin Substitute Fraud Detection

Graph-based machine learning model for detecting Medicare Part B billing fraud in skin substitute products (HCPCS Q4100-Q4397).

## 🎯 Purpose

This project demonstrates technical capability in fraud detection using Graph Neural Networks, supporting an NIW (National Interest Waiver) green card application. It implements the methodology described in academic papers like Yoo et al. (2023) "Medicare Fraud Detection Using Graph Analysis."

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/thatSandemaboy/skin-substitute-fraud-detection.git
cd skin-substitute-fraud-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data
python scripts/download_sample.py

# Build graph
python scripts/build_graph.py

# Train models
python scripts/train_model.py
```

## 📊 Results

**Current Status:** Working prototype with validated fraud detection

| Metric | Value |
|--------|-------|
| Providers analyzed | 557 |
| HCPCS codes covered | 59 |
| States represented | 47 |
| Graph nodes | 663 |
| Graph edges | 29,076 |
| Known fraud cases in sample | 1 |

### Validation Result

**Alexander Frank (OK)** — excluded by HHS-OIG Aug 2025 for false claims (1128a2)
- **Rank:** 93/557 (top 17%)
- **Detection:** ✅ Caught in top 100 investigation list
- **Combined Score:** 0.083 (heuristic: 0.087, GNN: 0.080)

### Top 10 High-Risk Providers (by combined score)

| Rank | Provider | State | Specialty | Services | Score |
|------|----------|-------|-----------|----------|-------|
| 1 | Ruth | CA | Orthopedic Surgery | 79 | 0.469 |
| 2 | Thome | MO | Orthopedic Surgery | 21 | 0.395 |
| 3 | Dorton | FL | Dermatology | 2,583 | 0.261 |
| 4 | Ting | CA | Dermatology | 14 | 0.253 |
| 5 | Javery | CO | Internal Medicine | 22,893 | 0.250 |
| 6 | Mitchell | CO | Family Practice | 28,948 | 0.242 |
| 7 | Nazarian | NY | Podiatry | 3,267 | 0.226 |
| 8 | Sandhu | FL | Dermatology | 16,591 | 0.220 |
| 9 | Stickler | FL | Dermatology | 3,918 | 0.219 |
| 10 | Christophersen | IA | Orthopedic Surgery | 44 | 0.210 |

## 🔬 Methodology

### 1. Data Collection
- Medicare Provider Utilization data (CMS API)
- LEIE exclusions database (HHS-OIG)
- Focus on skin substitute HCPCS codes (Q4100-Q4397)

### 2. Graph Construction
- **Nodes:** Providers (NPI), Products (HCPCS), States
- **Edges:** BILLED (provider→product), LOCATED_IN (provider→state), SIMILAR_PRODUCTS, SAME_STATE

### 3. Feature Engineering
- Tabular: Total services, beneficiaries, avg payment, charge-to-payment ratio
- Graph: Degree centrality, PageRank, clustering coefficient

### 4. Models
- **XGBoost Baseline:** Heuristic anomaly scoring (no labels)
- **Graph Autoencoder:** Unsupervised anomaly detection via reconstruction error

### 5. Key Insight
> Graph neural networks detect fraud *networks* (kickback rings, referral schemes) that traditional tabular analysis misses. This aligns with EO 14243's mandate to "eliminate information silos."

## 📁 Project Structure

```
├── data/
│   ├── processed/          # Processed data files
│   │   ├── skin_substitutes_sample.csv
│   │   ├── provider_graph.pkl
│   │   └── anomaly_results.csv
│   └── labels/
│       └── leie_exclusions.csv
├── models/
│   └── autoencoder.pt      # Trained GNN model
├── scripts/
│   ├── download_sample.py  # Data download via CMS API
│   ├── build_graph.py      # Graph construction
│   └── train_model.py      # Model training
├── src/models/
│   └── gnn.py             # GNN model definitions
└── docs/
    └── MODEL-EVALUATION-CRITERIA.md
```

## 📈 Success Metrics

| Level | Criteria | Status |
|-------|----------|--------|
| **Minimum** | Working GNN model | ✅ |
| **Good** | Identifies statistical outliers | ✅ |
| **Excellent** | Detects known excluded providers | ✅ (top 17%) |

### Detection Performance

With **zero labeled training data**, our unsupervised model:
- Detects the known fraud case in top 100 (of 557 providers)
- Achieves top 17% ranking for excluded provider
- Combines rule-based heuristics with graph structure learning

> **Note:** This is unsupervised learning — no fraud labels used during training. The model learns normal patterns and flags deviations.

## 🔑 Key References

- [OIG Skin Substitutes Report (Sept 2025)](https://oig.hhs.gov/reports/all/2025/medicare-part-b-payment-trends-for-skin-substitutes-raise-major-concerns-about-fraud-waste-and-abuse/)
- [DOJ $1.2B Fraud Sentencing](https://www.justice.gov/opa/pr/wound-graft-company-owners-sentenced-12b-health-care-fraud)
- EO 14243: "Stopping Waste, Fraud, and Abuse by Eliminating Information Silos"
- Yoo et al. (2023) "Medicare Fraud Detection Using Graph Analysis" — IEEE Access

## 🎯 NIW Alignment

This project demonstrates:
1. **Technical capability** in graph ML and healthcare data
2. **National benefit** through fraud detection methodology
3. **Alignment with EO 14243** on eliminating information silos
4. **Reproducible research** with open-source code

## 📝 License

MIT License - See LICENSE file for details.

## 👤 Author

Anthony Abavelim
- GitHub: [@thatSandemaboy](https://github.com/thatSandemaboy)
