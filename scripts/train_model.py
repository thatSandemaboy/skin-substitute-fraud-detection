#!/usr/bin/env python3
"""
Train fraud detection models on provider graph.

Implements:
1. XGBoost baseline (tabular, no graph)
2. GraphSAGE supervised (if labels available)
3. Graph Autoencoder unsupervised (anomaly detection)
"""

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Try to import ML libraries
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch/PyG not available. Running XGBoost baseline only.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available.")


def load_graph():
    """Load the provider graph."""
    graph_path = DATA_PROCESSED / "provider_graph.pkl"
    
    if not graph_path.exists():
        print("No graph found. Run build_graph.py first.")
        return None
    
    print(f"Loading graph from {graph_path.name}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def extract_features(G: nx.Graph):
    """Extract node features for ML models."""
    print("\nExtracting features...")
    
    # Get provider nodes
    providers = [(n, d) for n, d in G.nodes(data=True) 
                 if d.get('node_type') == 'provider']
    
    feature_names = [
        'total_services', 'total_beneficiaries', 'avg_payment',
        'avg_submitted_charge', 'charge_to_payment_ratio', 'num_products'
    ]
    
    features = []
    labels = []
    node_ids = []
    
    for node, data in providers:
        node_ids.append(node)
        
        feat = [
            data.get('total_services', 0),
            data.get('total_beneficiaries', 0),
            data.get('avg_payment', 0),
            data.get('avg_submitted_charge', 0),
            data.get('charge_to_payment_ratio', 0),
            data.get('num_products', 1),
        ]
        features.append(feat)
        labels.append(1 if data.get('is_excluded', False) else 0)
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    # Add graph-based features
    print("  Adding graph-based features...")
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    
    # PageRank (for importance)
    try:
        pagerank = nx.pagerank(G, max_iter=100)
    except:
        pagerank = {n: 0 for n in G.nodes()}
    
    # Clustering coefficient
    clustering = nx.clustering(G)
    
    graph_features = []
    for node in node_ids:
        graph_features.append([
            degree_cent.get(node, 0),
            pagerank.get(node, 0),
            clustering.get(node, 0),
        ])
    
    graph_features = np.array(graph_features, dtype=np.float32)
    X = np.hstack([X, graph_features])
    feature_names.extend(['degree_centrality', 'pagerank', 'clustering_coeff'])
    
    print(f"  Feature matrix: {X.shape}")
    print(f"  Fraud labels: {sum(labels)} / {len(labels)}")
    
    return X, y, node_ids, feature_names


def train_xgboost_baseline(X, y, feature_names):
    """Train XGBoost baseline (no graph structure)."""
    print("\n" + "=" * 60)
    print("Training XGBoost Baseline")
    print("=" * 60)
    
    if not XGB_AVAILABLE:
        print("XGBoost not available.")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # If we have labels, train supervised
    if sum(y) > 0:
        print("\nSupervised training with fraud labels...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=sum(y == 0) / max(sum(y == 1), 1),
            random_state=42,
            eval_metric='auc'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        print(f"\nTest Results:")
        print(f"  ROC AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        
        # Feature importance
        print("\nTop Features:")
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance.head(10).to_string(index=False))
        
        return model, scaler, auc
    
    else:
        print("\nNo fraud labels available. Using unsupervised anomaly detection...")
        
        # Use Isolation Forest-like approach via XGBoost
        # Train to predict random noise, high error = anomaly
        
        # Generate anomaly scores using feature-based heuristics
        anomaly_scores = compute_heuristic_anomaly_scores(X_scaled, feature_names)
        
        # Get top anomalies
        top_k = 20
        top_indices = np.argsort(anomaly_scores)[-top_k:][::-1]
        
        print(f"\nTop {top_k} Anomalous Providers (by heuristic score):")
        for i, idx in enumerate(top_indices[:10]):
            print(f"  {i+1}. Score: {anomaly_scores[idx]:.4f}")
        
        return None, scaler, anomaly_scores


def compute_heuristic_anomaly_scores(X, feature_names):
    """Compute anomaly scores based on statistical deviations."""
    scores = np.zeros(len(X))
    
    for i, name in enumerate(feature_names):
        col = X[:, i]
        mean = np.mean(col)
        std = np.std(col) + 1e-6
        
        # Z-score based anomaly
        z_scores = np.abs((col - mean) / std)
        
        # Weight certain features more
        weight = 1.0
        if 'charge_to_payment' in name:
            weight = 2.0  # High markup is suspicious
        elif 'total_services' in name:
            weight = 1.5  # High volume can be suspicious
        
        scores += z_scores * weight
    
    # Normalize
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    
    return scores


def train_graph_autoencoder(G, X, node_ids):
    """Train Graph Autoencoder for unsupervised anomaly detection."""
    print("\n" + "=" * 60)
    print("Training Graph Autoencoder (Unsupervised)")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available.")
        return None
    
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from models.gnn import AnomalyDetector
    
    # Create subgraph of just providers
    provider_subgraph = G.subgraph(node_ids).copy()
    
    # Convert to PyG format
    # Create node mapping
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    
    # Create edge index
    edges = []
    for u, v in provider_subgraph.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append([node_to_idx[u], node_to_idx[v]])
            edges.append([node_to_idx[v], node_to_idx[u]])  # Undirected
    
    if not edges:
        print("No edges in provider subgraph. Skipping GNN.")
        return None
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x = torch.tensor(X_scaled, dtype=torch.float)
    
    print(f"  Nodes: {len(node_ids)}, Edges: {edge_index.shape[1]//2}")
    print(f"  Features: {x.shape[1]}")
    
    # Create model
    model = AnomalyDetector(
        in_channels=x.shape[1],
        hidden_channels=32,
        latent_channels=16
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    # Training
    print("\nTraining...")
    model.train()
    epochs = 200
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        x_recon, z = model(x, edge_index)
        loss = F.mse_loss(x_recon, x)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Compute anomaly scores
    model.eval()
    with torch.no_grad():
        anomaly_scores = model.reconstruction_error(x, edge_index).numpy()
    
    print(f"\nAnomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    
    # Get top anomalies
    top_k = 20
    top_indices = np.argsort(anomaly_scores)[-top_k:][::-1]
    
    print(f"\nTop {top_k} Anomalous Providers (by reconstruction error):")
    for i, idx in enumerate(top_indices[:10]):
        node = node_ids[idx]
        data = G.nodes[node]
        print(f"  {i+1}. {data.get('name', 'Unknown')[:30]}")
        print(f"      Score: {anomaly_scores[idx]:.4f}, Services: {data.get('total_services', 0):.0f}")
    
    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    print(f"\n✅ Model saved to {MODELS_DIR / 'autoencoder.pt'}")
    
    return model, anomaly_scores, node_ids


def save_results(G, anomaly_scores, node_ids):
    """Save anomaly detection results."""
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    results = []
    for i, node in enumerate(node_ids):
        data = G.nodes[node]
        results.append({
            'npi': data.get('npi'),
            'name': data.get('name'),
            'state': data.get('state'),
            'provider_type': data.get('provider_type'),
            'total_services': data.get('total_services'),
            'total_beneficiaries': data.get('total_beneficiaries'),
            'avg_payment': data.get('avg_payment'),
            'charge_to_payment_ratio': data.get('charge_to_payment_ratio'),
            'num_products': data.get('num_products'),
            'anomaly_score': anomaly_scores[i] if i < len(anomaly_scores) else 0,
            'is_excluded': data.get('is_excluded', False),
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('anomaly_score', ascending=False)
    
    output_path = DATA_PROCESSED / "anomaly_results.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")
    
    # Print summary
    print("\nTop 20 Highest Risk Providers:")
    print(df[['name', 'state', 'total_services', 'anomaly_score']].head(20).to_string(index=False))
    
    return df


def main():
    """Train all models and evaluate."""
    print("=" * 60)
    print("Skin Substitute Fraud Detection - Model Training")
    print("=" * 60)
    
    # Load graph
    G = load_graph()
    if G is None:
        return
    
    # Extract features
    X, y, node_ids, feature_names = extract_features(G)
    
    # Train XGBoost baseline
    xgb_result = train_xgboost_baseline(X, y, feature_names)
    
    # Get anomaly scores
    if xgb_result and xgb_result[0] is None:
        # Using heuristic scores from XGBoost function
        anomaly_scores = xgb_result[2]
    else:
        anomaly_scores = np.zeros(len(node_ids))
    
    # Train Graph Autoencoder
    if TORCH_AVAILABLE:
        gnn_result = train_graph_autoencoder(G, X, node_ids)
        if gnn_result:
            _, gnn_scores, _ = gnn_result
            # Combine scores (average)
            anomaly_scores = (anomaly_scores + gnn_scores) / 2
    
    # Save results
    results_df = save_results(G, anomaly_scores, node_ids)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
    print("\nSummary:")
    print(f"  Total providers analyzed: {len(node_ids)}")
    print(f"  High-risk providers (score > 0.7): {sum(anomaly_scores > 0.7)}")
    print(f"  Medium-risk providers (0.5 < score <= 0.7): {sum((anomaly_scores > 0.5) & (anomaly_scores <= 0.7))}")
    
    if TORCH_AVAILABLE:
        print("\nNext steps:")
        print("  1. Review high-risk providers in anomaly_results.csv")
        print("  2. Cross-reference with OIG reports")
        print("  3. Validate findings against known fraud cases")


if __name__ == "__main__":
    main()
