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

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"


def load_data():
    """Load processed skin substitute data."""
    parquet_path = DATA_PROCESSED / "skin_substitutes_all.parquet"
    
    if not parquet_path.exists():
        print("No processed data found. Run filter_skin_substitutes.py first.")
        return None
    
    print(f"Loading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} claims")
    
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
        'Hcpcs_Cd': lambda x: list(x.unique()),
        'Rndrng_Prvdr_State_Abrvtn': 'first',
        'Rndrng_Prvdr_Type': 'first',
    }).reset_index()
    
    print(f"  {len(provider_stats):,} unique providers")
    
    # Add provider nodes
    for _, row in provider_stats.iterrows():
        npi = str(row['Rndrng_NPI'])
        
        is_excluded = npi in (excluded_npis or set())
        
        G.add_node(
            f"provider_{npi}",
            node_type="provider",
            npi=npi,
            total_services=row['Tot_Srvcs'],
            total_beneficiaries=row['Tot_Benes'],
            avg_payment=row['Avg_Mdcr_Pymt_Amt'],
            state=row['Rndrng_Prvdr_State_Abrvtn'],
            provider_type=row['Rndrng_Prvdr_Type'],
            is_excluded=is_excluded,  # Label!
        )
        
        # Add edges to products
        for hcpcs in row['Hcpcs_Cd']:
            product_node = f"product_{hcpcs}"
            if product_node not in G:
                G.add_node(product_node, node_type="product", hcpcs=hcpcs)
            G.add_edge(f"provider_{npi}", product_node, edge_type="BILLED")
        
        # Add edge to state
        state = row['Rndrng_Prvdr_State_Abrvtn']
        if state:
            state_node = f"state_{state}"
            if state_node not in G:
                G.add_node(state_node, node_type="state", state=state)
            G.add_edge(f"provider_{npi}", state_node, edge_type="LOCATED_IN")
    
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Count excluded providers
    excluded_count = sum(1 for n, d in G.nodes(data=True) 
                        if d.get('node_type') == 'provider' and d.get('is_excluded'))
    print(f"  Excluded (labeled) providers: {excluded_count:,}")
    
    return G


def add_provider_similarity_edges(G: nx.Graph, df: pd.DataFrame, threshold: float = 0.5):
    """Add edges between providers with similar billing patterns."""
    print("\nAdding provider similarity edges...")
    
    # Get provider-product matrix
    provider_products = df.groupby('Rndrng_NPI')['Hcpcs_Cd'].apply(set).to_dict()
    
    providers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'provider']
    
    edges_added = 0
    for i, p1 in enumerate(providers):
        npi1 = G.nodes[p1]['npi']
        prods1 = provider_products.get(int(npi1), set())
        
        for p2 in providers[i+1:]:
            npi2 = G.nodes[p2]['npi']
            prods2 = provider_products.get(int(npi2), set())
            
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
    
    # Add similarity edges (optional, can be slow)
    # G = add_provider_similarity_edges(G, df)
    
    # Save graph
    output_path = DATA_PROCESSED / "provider_graph.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"\n✅ Graph saved to {output_path}")
    
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
    
    print(f"\nTotal edges: {G.number_of_edges():,}")


if __name__ == "__main__":
    main()
