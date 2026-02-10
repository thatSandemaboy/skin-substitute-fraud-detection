#!/usr/bin/env python3
"""
Download 2021 and 2022 Medicare data for temporal backtest.
"""

import csv
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# CMS bulk data URLs
URLS = {
    "2022": "https://data.cms.gov/sites/default/files/2025-11/53fb2bae-4913-48dc-a6d4-d8c025906567/MUP_PHY_R25_P05_V20_D22_Prov_Svc.csv",
    "2021": "https://data.cms.gov/sites/default/files/2025-11/bffaf97a-c2ab-4fd7-8718-be90742e3485/MUP_PHY_R25_P05_V20_D21_Prov_Svc.csv",
}

def is_skin_substitute(hcpcs: str) -> bool:
    if not hcpcs:
        return False
    return hcpcs.upper().startswith('Q4')

def download_year(year: str, url: str):
    output = DATA_DIR / f"skin_substitutes_{year}.csv"
    if output.exists():
        print(f"✅ {year} already exists: {output}")
        return
    
    print(f"Downloading {year} data...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    lines = response.iter_lines(decode_unicode=True)
    header = next(lines)
    reader_header = next(csv.reader([header]))
    
    hcpcs_idx = None
    for i, col in enumerate(reader_header):
        if 'HCPCS' in col.upper() and 'CD' in col.upper():
            hcpcs_idx = i
            break
    
    matched = 0
    total = 0
    
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(reader_header)
        
        for line in tqdm(lines, desc=f"Processing {year}"):
            total += 1
            try:
                row = next(csv.reader([line]))
                if len(row) > hcpcs_idx and is_skin_substitute(row[hcpcs_idx]):
                    writer.writerow(row)
                    matched += 1
            except:
                pass
    
    print(f"✅ {year}: {matched:,} skin substitute records from {total:,} total")

def main():
    for year, url in URLS.items():
        download_year(year, url)

if __name__ == "__main__":
    main()
