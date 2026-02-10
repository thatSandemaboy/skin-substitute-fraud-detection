#!/usr/bin/env python3
"""
Build provider graph from skin substitute claims data.

Graph Structure:
- Nodes: Providers (NPI), Products (HCPCS), Locations (State)
- Edges: BILLED (provider -> product), LOCATED_IN (provider -> state)
"""

import pandas as pd
import networkx as nx
from pathlib import Path
import pickle
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"


def load_data():
    """Load processed skin substitute data."""
    # Try sample first, then full dataset
    csv_path = DATA_PROCESSED / "skin_substitutes_sample.csv"
    if not csv_path.exists():
        csv_path = DATA_PROCESSED / "skin_substitutes_all_years.csv"
    
    if not csv_path.exists():
        print("No processed data found. Run download_sample.py first.")
        return None
    
    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} claims")
    
    # Convert numeric columns
    numeric_cols = ['Tot_Srvcs', 'Tot_Benes', 'Avg_Mdcr_Pymt_Amt', 'Avg_Sbmtd_Chrg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def load_leie_labels():
    """Load LEIE exclusion labels."""
    leie_path = DATA_LABELS / "leie_exclusions.csv"
    
    if not leie_path.exists():
        print("No LEIE data found. Run download_data.py first.")
        return set()
    
    print("Loading LEIE exclusions...")
    leie = pd.read_csv(leie_path, low_memory=False)
    
    # Extract NPIs (column may vary)
    npi_col = [c for c in leie.columns if 'NPI' in c.upper()]
    if npi_col:
        excluded_npis = set(leie[npi_col[0]].dropna().astype(str))
        # Clean up NPIs (remove leading zeros, etc.)
        excluded_npis = {npi.lstrip('0') for npi in excluded_npis if npi != '0000000000'}
        print(f"  {len(excluded_npis):,} excluded NPIs")
        return excluded_npis
    
    return set()


def build_provider_graph(df: pd.DataFrame, excluded_npis: set = None):
    """Build a heterogeneous graph from claims data."""
    print("\nBuilding provider graph...")
    
    G = nx.Graph()
    
    # Aggregate by provider
    provider_stats = df.groupby('Rndrng_NPI').agg({
        'Tot_Srvcs': 'sum',
        'Tot_Benes': 'sum', 
        'Avg_Mdcr_Pymt_Amt': 'mean',
        'Avg_Sbmtd_Chrg': 'mean',
        'HCPCS_Cd': lambda x: list(x.unique()),
        'Rndrng_Prvdr_State_Abrvtn': 'first',
        'Rndrng_Prvdr_Type': 'first',
        'Rndrng_Prvdr_Last_Org_Name': 'first',
    }).reset_index()
    
    print(f"  {len(provider_stats):,} unique providers")
    
    # Add provider nodes
    for _, row in provider_stats.iterrows():
        npi = str(row['Rndrng_NPI'])
        
        is_excluded = npi in (excluded_npis or set()) or npi.lstrip('0') in (excluded_npis or set())
        
        # Calculate anomaly features
        charge_to_payment_ratio = (
            row['Avg_Sbmtd_Chrg'] / row['Avg_Mdcr_Pymt_Amt'] 
            if row['Avg_Mdcr_Pymt_Amt'] > 0 else 0
        )
        
        G.add_node(
            f"provider_{npi}",
            node_type="provider",
            npi=npi,
            name=row['Rndrng_Prvdr_Last_Org_Name'],
            total_services=float(row['Tot_Srvcs']),
            total_beneficiaries=float(row['Tot_Benes']),
            avg_payment=float(row['Avg_Mdcr_Pymt_Amt']),
            avg_submitted_charge=float(row['Avg_Sbmtd_Chrg']),
            charge_to_payment_ratio=float(charge_to_payment_ratio),
            state=row['Rndrng_Prvdr_State_Abrvtn'],
            provider_type=row['Rndrng_Prvdr_Type'],
            num_products=len(row['HCPCS_Cd']),
            is_excluded=is_excluded,  # Label!
        )
        
        # Add edges to products
        for hcpcs in row['HCPCS_Cd']:
            product_node = f"product_{hcpcs}"
            if product_node not in G:
                G.add_node(product_node, node_type="product", hcpcs=hcpcs)
            G.add_edge(f"provider_{npi}", product_node, edge_type="BILLED")
        
        # Add edge to state
        state = row['Rndrng_Prvdr_State_Abrvtn']
        if state and pd.notna(state):
            state_node = f"state_{state}"
            if state_node not in G:
                G.add_node(state_node, node_type="state", state=state)
            G.add_edge(f"provider_{npi}", state_node, edge_type="LOCATED_IN")
    
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Count excluded providers
    excluded_count = sum(1 for n, d in G.nodes(data=True) 
                        if d.get('node_type') == 'provider' and d.get('is_excluded'))
    print(f"  Excluded (labeled fraud) providers: {excluded_count:,}")
    
    return G


def add_provider_similarity_edges(G: nx.Graph, threshold: float = 0.3):
    """Add edges between providers with similar billing patterns."""
    print("\nAdding provider similarity edges...")
    
    providers = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'provider']
    
    # Build provider-product sets
    provider_products = {}
    for n, d in providers:
        products = set()
        for neighbor in G.neighbors(n):
            if G.nodes[neighbor].get('node_type') == 'product':
                products.add(neighbor)
        provider_products[n] = products
    
    edges_added = 0
    for i, (p1, d1) in enumerate(providers):
        prods1 = provider_products[p1]
        
        for p2, d2 in providers[i+1:]:
            prods2 = provider_products[p2]
            
            # Jaccard similarity
            if prods1 and prods2:
                intersection = len(prods1 & prods2)
                union = len(prods1 | prods2)
                similarity = intersection / union
                
                if similarity >= threshold:
                    G.add_edge(p1, p2, edge_type="SIMILAR_PRODUCTS", weight=similarity)
                    edges_added += 1
    
    print(f"  Added {edges_added:,} similarity edges")
    return G


def add_geographic_edges(G: nx.Graph):
    """Add edges between providers in the same state."""
    print("\nAdding geographic co-location edges...")
    
    providers = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'provider']
    
    # Group by state
    state_providers = {}
    for n, d in providers:
        state = d.get('state')
        if state:
            if state not in state_providers:
                state_providers[state] = []
            state_providers[state].append(n)
    
    edges_added = 0
    for state, provs in state_providers.items():
        # Connect providers in same state (for small states)
        if len(provs) <= 10:
            for i, p1 in enumerate(provs):
                for p2 in provs[i+1:]:
                    if not G.has_edge(p1, p2):
                        G.add_edge(p1, p2, edge_type="SAME_STATE", weight=0.5)
                        edges_added += 1
    
    print(f"  Added {edges_added:,} geographic edges")
    return G


def main():
    """Build and save the provider graph."""
    print("=" * 60)
    print("Building Provider Graph")
    print("=" * 60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    excluded_npis = load_leie_labels()
    
    # Build graph
    G = build_provider_graph(df, excluded_npis)
    
    # Add similarity edges
    G = add_provider_similarity_edges(G, threshold=0.3)
    
    # Add geographic edges
    G = add_geographic_edges(G)
    
    # Save graph
    output_path = DATA_PROCESSED / "provider_graph.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"\n✅ Graph saved to {output_path}")
    
    # Also save in GML format for visualization
    gml_path = DATA_PROCESSED / "provider_graph.gml"
    nx.write_gml(G, gml_path)
    print(f"✅ Graph saved to {gml_path}")
    
    # Summary stats
    print("\n" + "=" * 60)
    print("Graph Summary")
    print("=" * 60)
    
    node_types = {}
    for n, d in G.nodes(data=True):
        t = d.get('node_type', 'unknown')
        node_types[t] = node_types.get(t, 0) + 1
    
    for t, count in sorted(node_types.items()):
        print(f"  {t}: {count:,}")
    
    edge_types = {}
    for u, v, d in G.edges(data=True):
        t = d.get('edge_type', 'unknown')
        edge_types[t] = edge_types.get(t, 0) + 1
    
    print(f"\nEdge types:")
    for t, count in sorted(edge_types.items()):
        print(f"  {t}: {count:,}")
    
    print(f"\nTotal nodes: {G.number_of_nodes():,}")
    print(f"Total edges: {G.number_of_edges():,}")
    
    # Check for labeled fraud cases
    fraud_count = sum(1 for n, d in G.nodes(data=True) 
                     if d.get('node_type') == 'provider' and d.get('is_excluded'))
    print(f"\nLabeled fraud cases: {fraud_count}")


if __name__ == "__main__":
    main()
