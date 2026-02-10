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

**Current Status:** Working prototype with sample data

| Metric | Value |
|--------|-------|
| Providers analyzed | 229 |
| HCPCS codes covered | 45 |
| States represented | 40 |
| Graph nodes | 314 |
| Graph edges | 5,856 |

### Top Anomalous Providers (by combined score)

| Provider | State | Services | Anomaly Score |
|----------|-------|----------|---------------|
| Soleymani | IN | 49 | 0.50 |
| Sandhu | FL | 16,591 | 0.49 |
| Meo | NY | 15 | 0.45 |
| Nazarian | NY | 3,267 | 0.34 |
| Ellington | TX | 12,102 | 0.32 |

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
| **Excellent** | Flags known DOJ cases | 🔄 Pending validation |

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
