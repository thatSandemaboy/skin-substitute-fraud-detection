# Skin Substitute Fraud Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Graph-based fraud detection for Medicare Part B skin substitute billing.**

Detects kickback rings, billing anomalies, and suspicious provider networks using Graph Neural Networks (GNNs) on public CMS data.

## 🔥 Why This Matters

- **$10B+ annually** in Medicare skin substitute spending (2024)
- **$1.2B fraud case** — DOJ's largest skin substitute prosecution (2025)
- **90% payment cut** — CMS slashed reimbursements Jan 2026 due to rampant fraud

This project applies graph machine learning to detect fraud patterns that traditional tabular methods miss — specifically **network fraud** like kickback rings and coordinated billing schemes.

## 🎯 What It Does

1. **Data Pipeline** — Downloads and processes public Medicare Part B data
2. **Graph Construction** — Builds provider-product-location networks
3. **Anomaly Detection** — Identifies suspicious billing patterns
4. **Ring Detection** — Finds coordinated fraud networks using GNNs
5. **Explainability** — Generates human-readable explanations for flagged cases

## 📊 Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Medicare Provider Utilization | [CMS](https://data.cms.gov) | Provider billing records |
| LEIE Exclusions | [HHS-OIG](https://oig.hhs.gov/exclusions/) | Confirmed fraud cases (labels) |
| HCPCS Q4100-Q4397 | CMS | Skin substitute procedure codes |

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/thatSandemaboy/skin-substitute-fraud-detection.git
cd skin-substitute-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Run analysis
python scripts/build_graph.py
python scripts/train_model.py
```

## 📁 Project Structure

```
├── data/
│   ├── raw/              # Downloaded CMS data
│   ├── processed/        # Cleaned, graph-ready data
│   └── labels/           # LEIE exclusion labels
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_gnn_training.ipynb
├── src/
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature engineering
│   ├── models/           # GNN model definitions
│   └── visualization/    # Graph visualization
├── scripts/
│   ├── download_data.py
│   ├── build_graph.py
│   └── train_model.py
├── tests/
├── requirements.txt
└── README.md
```

## 🧠 Methodology

### Graph Structure

```
Provider ──[BILLED]──> Product (HCPCS)
    │                      │
    └──[REFERRED_TO]──> Provider
    │
    └──[LOCATED_IN]──> Location
```

### Detection Approaches

1. **Supervised Classification** — Predict LEIE exclusion using GraphSAGE
2. **Anomaly Detection** — Graph autoencoders for outlier detection
3. **Community Detection** — Find suspicious provider clusters

## 📚 References

- [OIG Report: Skin Substitutes FWA (Sept 2025)](https://oig.hhs.gov/reports/all/2025/medicare-part-b-payment-trends-for-skin-substitutes-raise-major-concerns-about-fraud-waste-and-abuse/)
- [DOJ: $1.2B Skin Substitute Fraud Sentencing](https://www.justice.gov/opa/pr/wound-graft-company-owners-sentenced-12b-health-care-fraud-and-agree-pay-309m-resolve-civil)
- [EO 14243: Stopping Waste, Fraud, and Abuse](https://www.whitehouse.gov/presidential-actions/2025/03/stopping-waste-fraud-and-abuse-by-eliminating-information-silos/)
- Yoo et al. (2023) "Medicare Fraud Detection Using Graph Analysis" — IEEE Access

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

*Built with the goal of supporting federal program integrity efforts and advancing open-source fraud detection research.*
