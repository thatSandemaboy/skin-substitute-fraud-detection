#!/usr/bin/env python3
"""
Filter Medicare data to skin substitute claims only (HCPCS Q4100-Q4397).
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Skin substitute HCPCS codes
SKIN_SUB_CODES = set(f"Q4{i:03d}" for i in range(100, 398))


def filter_medicare_file(input_path: Path, output_path: Path):
    """Filter a Medicare CSV to skin substitute codes only."""
    print(f"\nProcessing {input_path.name}...")
    
    # Read in chunks (files are ~2GB)
    chunks = []
    chunk_size = 500_000
    
    for chunk in tqdm(pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)):
        # Filter to skin substitute codes
        mask = chunk['Hcpcs_Cd'].isin(SKIN_SUB_CODES)
        filtered = chunk[mask]
        
        if len(filtered) > 0:
            chunks.append(filtered)
    
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        print(f"  Found {len(df):,} skin substitute claims")
        
        # Save
        df.to_parquet(output_path, index=False)
        print(f"  Saved to {output_path.name}")
        
        return df
    else:
        print("  No skin substitute claims found!")
        return None


def main():
    """Filter all Medicare files to skin substitutes."""
    print("=" * 60)
    print("Filtering Medicare Data to Skin Substitutes")
    print("=" * 60)
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Find all Medicare CSV files
    csv_files = list(DATA_RAW.glob("Medicare_Provider_Util*.csv"))
    
    if not csv_files:
        print("No Medicare CSV files found in data/raw/")
        print("Run `python scripts/download_data.py` first")
        return
    
    all_dfs = []
    for csv_file in csv_files:
        year = csv_file.stem.split("CY")[-1][:4]
        output_path = DATA_PROCESSED / f"skin_substitutes_{year}.parquet"
        
        df = filter_medicare_file(csv_file, output_path)
        if df is not None:
            df['year'] = int(year)
            all_dfs.append(df)
    
    # Combine all years
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(DATA_PROCESSED / "skin_substitutes_all.parquet", index=False)
        
        print("\n" + "=" * 60)
        print(f"✅ Combined dataset: {len(combined):,} claims")
        print(f"   Unique providers: {combined['Rndrng_NPI'].nunique():,}")
        print(f"   Unique products: {combined['Hcpcs_Cd'].nunique():,}")
        print("=" * 60)


if __name__ == "__main__":
    main()
