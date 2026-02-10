#!/usr/bin/env python3
"""
Download a sample of skin substitute data for prototyping.

Strategy: Sample the API to get ~200+ skin substitute records.
This is enough to build a working prototype and prove the concept.
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import random

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# CMS API configuration
DATASET_ID = "92396110-2aed-4d63-a6a2-5d6207d46a29"
API_BASE = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_ID}/data"

# Skin substitute HCPCS codes (Q4100-Q4397)
SKIN_SUB_PREFIX = "Q4"


def sample_skin_substitutes(target_records=300, max_api_calls=200):
    """
    Sample skin substitute records using random offsets.
    This is faster than scanning sequentially.
    """
    print("\n📥 Sampling skin substitute records from CMS API...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    output_file = DATA_PROCESSED / "skin_substitutes_sample.csv"
    
    all_records = []
    page_size = 5000
    
    # Estimate dataset size (~10M records) and sample random chunks
    estimated_size = 10_000_000
    
    pbar = tqdm(total=target_records, desc="Finding records")
    
    api_calls = 0
    while len(all_records) < target_records and api_calls < max_api_calls:
        try:
            # Random offset to sample different parts of the dataset
            offset = random.randint(0, estimated_size - page_size)
            
            url = f"{API_BASE}?size={page_size}&offset={offset}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            api_calls += 1
            
            if not data:
                continue
            
            # Filter for skin substitute codes (Q4xxx)
            skin_sub_records = [
                r for r in data 
                if r.get('HCPCS_Cd', '').startswith(SKIN_SUB_PREFIX)
            ]
            
            # Avoid duplicates
            existing_keys = {(r['Rndrng_NPI'], r['HCPCS_Cd']) for r in all_records}
            new_records = [
                r for r in skin_sub_records 
                if (r['Rndrng_NPI'], r['HCPCS_Cd']) not in existing_keys
            ]
            
            all_records.extend(new_records)
            pbar.update(len(new_records))
            pbar.set_postfix({
                'calls': api_calls, 
                'found': len(all_records)
            })
            
            # Rate limiting
            time.sleep(0.3)
            
        except Exception as e:
            print(f"\n  Error: {e}")
            time.sleep(2)
            continue
    
    pbar.close()
    
    print(f"\n  Made {api_calls} API calls")
    print(f"  Found {len(all_records)} skin substitute records")
    
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Unique providers: {df['Rndrng_NPI'].nunique():,}")
        print(f"    Unique HCPCS codes: {df['HCPCS_Cd'].nunique()}")
        print(f"    States: {df['Rndrng_Prvdr_State_Abrvtn'].nunique()}")
        
        # Show top codes
        print(f"\n  Top 10 HCPCS codes:")
        for code, count in df['HCPCS_Cd'].value_counts().head(10).items():
            desc = df[df['HCPCS_Cd'] == code]['HCPCS_Desc'].iloc[0][:50]
            print(f"    {code}: {count} records - {desc}...")
        
        return output_file
    else:
        print("  ⚠️ No skin substitute records found")
        return None


def main():
    """Download sample data for prototyping."""
    print("=" * 60)
    print("Skin Substitute Fraud Detection - Sample Download")
    print("=" * 60)
    print("\nSampling ~300 skin substitute records for prototyping.")
    
    # Create directories
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Sample skin substitute records
    sample_skin_substitutes(target_records=300, max_api_calls=200)
    
    print("\n" + "=" * 60)
    print("✅ Sample download complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. python scripts/build_graph.py")
    print("  2. python scripts/train_model.py")


if __name__ == "__main__":
    main()
