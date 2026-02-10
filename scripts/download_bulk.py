#!/usr/bin/env python3
"""
Stream download Medicare data and filter to skin substitutes (Q4xxx codes).
This avoids storing the full 4GB file.
"""

import csv
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "processed" / "skin_substitutes_full.csv"

# CMS bulk data URL (2023)
URL = "https://data.cms.gov/sites/default/files/2025-04/e3f823f8-db5b-4cc7-ba04-e7ae92b99757/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv"

def is_skin_substitute(hcpcs: str) -> bool:
    """Check if HCPCS code is skin substitute (Q4xxx)."""
    if not hcpcs:
        return False
    return hcpcs.upper().startswith('Q4')

def main():
    print(f"Downloading and filtering: {URL}")
    print("This may take 30-60 minutes...\n")
    
    response = requests.get(URL, stream=True)
    response.raise_for_status()
    
    # Track progress by bytes
    total_bytes = 0
    matched_rows = 0
    total_rows = 0
    
    lines = response.iter_lines(decode_unicode=True)
    header = next(lines)
    
    # Parse header to find HCPCS column
    reader_header = next(csv.reader([header]))
    hcpcs_idx = None
    for i, col in enumerate(reader_header):
        if 'HCPCS' in col.upper() and 'CD' in col.upper():
            hcpcs_idx = i
            break
    
    if hcpcs_idx is None:
        print("ERROR: Could not find HCPCS_Cd column")
        print(f"Columns: {reader_header}")
        return
    
    print(f"Found HCPCS column at index {hcpcs_idx}")
    
    with open(OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(reader_header)
        
        pbar = tqdm(lines, desc="Processing", unit=" rows")
        for line in pbar:
            total_rows += 1
            total_bytes += len(line)
            
            try:
                row = next(csv.reader([line]))
                if len(row) > hcpcs_idx:
                    hcpcs = row[hcpcs_idx]
                    if is_skin_substitute(hcpcs):
                        writer.writerow(row)
                        matched_rows += 1
            except:
                pass
            
            if total_rows % 100000 == 0:
                pbar.set_postfix({
                    'matched': matched_rows,
                    'MB': f"{total_bytes/1024/1024:.1f}"
                })
    
    print(f"\n✅ Done!")
    print(f"   Total rows processed: {total_rows:,}")
    print(f"   Skin substitute rows: {matched_rows:,}")
    print(f"   Saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
