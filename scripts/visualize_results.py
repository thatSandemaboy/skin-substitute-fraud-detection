#!/usr/bin/env python3
"""
Visualize fraud detection results.
"""

import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_data():
    """Load graph and results."""
    graph_path = DATA_PROCESSED / "provider_graph.pkl"
    results_path = DATA_PROCESSED / "anomaly_results.csv"
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    results = pd.read_csv(results_path)
    
    return G, results


def plot_anomaly_distribution(results):
    """Plot distribution of anomaly scores."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(results['anomaly_score'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.3, color='orange', linestyle='--', label='Medium Risk Threshold')
    plt.axvline(x=0.5, color='red', linestyle='--', label='High Risk Threshold')
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Number of Providers', fontsize=12)
    plt.title('Distribution of Fraud Risk Scores', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'anomaly_distribution.png', dpi=150)
    plt.close()
    print("✅ Saved anomaly_distribution.png")


def plot_services_vs_score(results):
    """Plot services vs anomaly score."""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(
        results['total_services'], 
        results['anomaly_score'],
        alpha=0.6,
        c=results['anomaly_score'],
        cmap='RdYlGn_r'
    )
    
    plt.xlabel('Total Services', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title('Provider Services vs Fraud Risk Score', fontsize=14)
    plt.colorbar(label='Risk Score')
    
    # Annotate top outliers
    top_5 = results.nlargest(5, 'anomaly_score')
    for _, row in top_5.iterrows():
        plt.annotate(
            row['name'][:15],
            (row['total_services'], row['anomaly_score']),
            fontsize=8,
            alpha=0.8
        )
    
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'services_vs_score.png', dpi=150)
    plt.close()
    print("✅ Saved services_vs_score.png")


def plot_state_heatmap(results):
    """Plot anomaly by state."""
    plt.figure(figsize=(12, 6))
    
    state_scores = results.groupby('state').agg({
        'anomaly_score': 'mean',
        'npi': 'count'
    }).rename(columns={'npi': 'provider_count'})
    
    state_scores = state_scores.sort_values('anomaly_score', ascending=False)
    
    colors = plt.cm.RdYlGn_r(state_scores['anomaly_score'] / state_scores['anomaly_score'].max())
    
    bars = plt.bar(
        range(len(state_scores)),
        state_scores['anomaly_score'],
        color=colors
    )
    
    plt.xticks(range(len(state_scores)), state_scores.index, rotation=45, ha='right')
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Average Anomaly Score', fontsize=12)
    plt.title('Average Fraud Risk Score by State', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'state_heatmap.png', dpi=150)
    plt.close()
    print("✅ Saved state_heatmap.png")


def plot_graph_visualization(G, results):
    """Create network visualization."""
    plt.figure(figsize=(14, 10))
    
    # Get provider nodes only
    providers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'provider']
    subgraph = G.subgraph(providers)
    
    # Create score mapping
    score_map = dict(zip(
        ['provider_' + str(int(npi)) for npi in results['npi']],
        results['anomaly_score']
    ))
    
    # Node colors based on anomaly score
    node_colors = [score_map.get(n, 0) for n in subgraph.nodes()]
    
    # Node sizes based on services
    size_map = dict(zip(
        ['provider_' + str(int(npi)) for npi in results['npi']],
        np.log1p(results['total_services']) * 50
    ))
    node_sizes = [size_map.get(n, 100) for n in subgraph.nodes()]
    
    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.RdYlGn_r,
        alpha=0.7
    )
    
    nx.draw_networkx_edges(
        subgraph, pos,
        alpha=0.1,
        edge_color='gray'
    )
    
    plt.title('Provider Network (color=risk, size=volume)', fontsize=14)
    plt.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn_r,
        norm=plt.Normalize(vmin=0, vmax=max(node_colors) if node_colors else 1)
    )
    sm.set_array([])
    cbar = plt.gcf().colorbar(sm, ax=plt.gca(), label='Anomaly Score', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'network_visualization.png', dpi=150)
    plt.close()
    print("✅ Saved network_visualization.png")


def create_summary_table(results):
    """Create summary statistics table."""
    summary = {
        'Total Providers': len(results),
        'Mean Anomaly Score': f"{results['anomaly_score'].mean():.4f}",
        'Max Anomaly Score': f"{results['anomaly_score'].max():.4f}",
        'High Risk (>0.5)': (results['anomaly_score'] > 0.5).sum(),
        'Medium Risk (0.3-0.5)': ((results['anomaly_score'] > 0.3) & (results['anomaly_score'] <= 0.5)).sum(),
        'Total Services': f"{results['total_services'].sum():,.0f}",
        'Unique States': results['state'].nunique(),
    }
    
    summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
    summary_df.to_csv(FIGURES_DIR / 'summary_statistics.csv', index=False)
    
    print("\n📊 Summary Statistics:")
    print(summary_df.to_string(index=False))
    print("✅ Saved summary_statistics.csv")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    FIGURES_DIR.mkdir(exist_ok=True)
    
    G, results = load_data()
    
    print(f"\nLoaded {len(results)} providers")
    
    plot_anomaly_distribution(results)
    plot_services_vs_score(results)
    plot_state_heatmap(results)
    plot_graph_visualization(G, results)
    create_summary_table(results)
    
    print("\n" + "=" * 60)
    print("✅ All visualizations saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
