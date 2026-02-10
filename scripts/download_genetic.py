#!/usr/bin/env python3
"""
Download Medicare data filtered to genetic testing codes (81xxx).
"""

import csv
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "processed" / "genetic_testing_2023.csv"

# CMS bulk data URL (2023)
URL = "https://data.cms.gov/sites/default/files/2025-04/e3f823f8-db5b-4cc7-ba04-e7ae92b99757/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv"

def is_genetic_test(hcpcs: str) -> bool:
    """Check if HCPCS code is genetic/molecular pathology (81xxx)."""
    if not hcpcs:
        return False
    return hcpcs.startswith('81')

def main():
    print(f"Downloading genetic testing data (81xxx codes)...")
    print("This may take 5-10 minutes...\n")
    
    response = requests.get(URL, stream=True)
    response.raise_for_status()
    
    lines = response.iter_lines(decode_unicode=True)
    header = next(lines)
    reader_header = next(csv.reader([header]))
    
    hcpcs_idx = None
    for i, col in enumerate(reader_header):
        if 'HCPCS' in col.upper() and 'CD' in col.upper():
            hcpcs_idx = i
            break
    
    print(f"Found HCPCS column at index {hcpcs_idx}")
    
    matched = 0
    total = 0
    
    with open(OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(reader_header)
        
        for line in tqdm(lines, desc="Processing"):
            total += 1
            try:
                row = next(csv.reader([line]))
                if len(row) > hcpcs_idx and is_genetic_test(row[hcpcs_idx]):
                    writer.writerow(row)
                    matched += 1
            except:
                pass
            
            if total % 500000 == 0:
                print(f"  {total:,} processed, {matched:,} genetic tests found")
    
    print(f"\n✅ Done!")
    print(f"   Total rows processed: {total:,}")
    print(f"   Genetic testing rows: {matched:,}")
    print(f"   Saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
