#!/usr/bin/env python3
"""
Download skin substitute data from CMS API.

This is much more efficient than downloading the full 4GB+ dataset.
It queries the API and filters for Q4100-Q4397 codes (skin substitutes).
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# CMS API configuration
# Dataset: Medicare Physician & Other Practitioners - by Provider and Service
DATASET_ID = "92396110-2aed-4d63-a6a2-5d6207d46a29"
API_BASE = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_ID}/data"

# Skin substitute HCPCS codes (Q4100-Q4397)
SKIN_SUB_PREFIX = "Q4"

# LEIE download URL
LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"


def download_leie():
    """Download LEIE exclusions database."""
    print("\n📥 Downloading LEIE exclusions database...")
    DATA_LABELS.mkdir(parents=True, exist_ok=True)
    
    dest = DATA_LABELS / "leie_exclusions.csv"
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return dest
    
    response = requests.get(LEIE_URL, stream=True)
    response.raise_for_status()
    
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"  ✓ LEIE ready: {dest}")
    return dest


def get_total_records():
    """Get total number of records in the dataset."""
    # CMS API doesn't easily expose total count, so we'll paginate
    return None


def download_skin_substitutes():
    """Download skin substitute records from CMS API."""
    print("\n📥 Downloading skin substitute records from CMS API...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    output_file = DATA_PROCESSED / "skin_substitutes_all_years.csv"
    
    all_records = []
    page_size = 5000  # Max per request
    offset = 0
    total_scanned = 0
    
    # We'll scan through all records and filter for Q4 codes
    # This is more reliable than trusting API filters
    
    pbar = tqdm(desc="Scanning records", unit=" records")
    
    while True:
        try:
            url = f"{API_BASE}?size={page_size}&offset={offset}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                break
            
            # Filter for skin substitute codes (Q4xxx)
            skin_sub_records = [
                r for r in data 
                if r.get('HCPCS_Cd', '').startswith(SKIN_SUB_PREFIX)
            ]
            
            all_records.extend(skin_sub_records)
            total_scanned += len(data)
            
            pbar.update(len(data))
            pbar.set_postfix({'found': len(all_records)})
            
            if len(data) < page_size:
                break
            
            offset += page_size
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n  Error at offset {offset}: {e}")
            time.sleep(5)
            continue
    
    pbar.close()
    
    print(f"\n  Scanned {total_scanned:,} total records")
    print(f"  Found {len(all_records):,} skin substitute records")
    
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Unique providers: {df['Rndrng_NPI'].nunique():,}")
        print(f"    Unique HCPCS codes: {df['HCPCS_Cd'].nunique()}")
        print(f"    Date range: Latest data (2023)")
        
        return output_file
    else:
        print("  ⚠️ No skin substitute records found")
        return None


def check_for_historical_data():
    """Check if historical endpoints exist."""
    print("\n🔍 Checking for historical data endpoints...")
    
    # Try to find historical dataset versions
    data_json_url = "https://data.cms.gov/data.json"
    response = requests.get(data_json_url)
    data = response.json()
    
    historical = []
    for dataset in data.get('dataset', []):
        title = dataset.get('title', '')
        if 'Medicare Physician' in title and 'Provider and Service' in title:
            historical.append({
                'title': title,
                'identifier': dataset.get('identifier'),
                'temporal': dataset.get('temporal'),
                'modified': dataset.get('modified')
            })
    
    print(f"  Found {len(historical)} related datasets:")
    for h in historical:
        print(f"    - {h['title']}")
        print(f"      Temporal: {h.get('temporal')}")
    
    return historical


def main():
    """Download all required data."""
    print("=" * 60)
    print("Skin Substitute Fraud Detection - Smart Data Download")
    print("=" * 60)
    print("\nUsing CMS API to download only skin substitute records.")
    print("This is much faster than downloading the full 4GB+ dataset!")
    
    # Create directories
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_LABELS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Download LEIE (fraud labels)
    download_leie()
    
    # Check what historical data is available
    check_for_historical_data()
    
    # Download skin substitute records
    download_skin_substitutes()
    
    print("\n" + "=" * 60)
    print("✅ Data download complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. python scripts/build_graph.py")
    print("  2. python scripts/train_model.py")


if __name__ == "__main__":
    main()
