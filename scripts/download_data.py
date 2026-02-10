#!/usr/bin/env python3
"""
Download Medicare Part B and LEIE data for skin substitute fraud detection.

Data Sources:
- Medicare Provider Utilization and Payment Data (CMS)
- List of Excluded Individuals/Entities (HHS-OIG)
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import gzip
import shutil

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"

# Data URLs
URLS = {
    # Medicare Provider Utilization - Physician and Other Suppliers
    # Filter to skin substitutes (Q4100-Q4397) after download
    "medicare_2022": "https://data.cms.gov/sites/default/files/2024-04/67d6ab35-cf5d-4f32-b5c6-a54fa8936fd2/Medicare_Provider_Util_Payment_PUF_CY2022.csv.zip",
    "medicare_2021": "https://data.cms.gov/sites/default/files/2023-04/7ef60ed6-5cc7-4a7e-9f36-7bda5f86336b/Medicare_Provider_Util_Payment_PUF_CY2021.csv.zip",
    
    # LEIE - List of Excluded Individuals/Entities (fraud labels)
    "leie": "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv",
}

# Skin substitute HCPCS codes (Q4100-Q4397)
SKIN_SUB_CODES = [f"Q4{i:03d}" for i in range(100, 398)]


def download_file(url: str, dest: Path, desc: str = None) -> Path:
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return dest
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc=desc or dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    return dest_dir


def download_medicare_data(year: str = "2022"):
    """Download Medicare Provider Utilization data."""
    print(f"\n📥 Downloading Medicare {year} data...")
    
    url = URLS.get(f"medicare_{year}")
    if not url:
        print(f"  No URL for year {year}")
        return
    
    zip_path = DATA_RAW / f"medicare_{year}.csv.zip"
    download_file(url, zip_path, f"Medicare {year}")
    
    # Extract
    extract_zip(zip_path, DATA_RAW)
    print(f"  ✓ Medicare {year} ready")


def download_leie():
    """Download LEIE exclusions database."""
    print("\n📥 Downloading LEIE exclusions database...")
    
    dest = DATA_LABELS / "leie_exclusions.csv"
    download_file(URLS["leie"], dest, "LEIE Database")
    print("  ✓ LEIE ready")


def main():
    """Download all required data."""
    print("=" * 60)
    print("Skin Substitute Fraud Detection - Data Download")
    print("=" * 60)
    
    # Create directories
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_LABELS.mkdir(parents=True, exist_ok=True)
    
    # Download data
    download_leie()
    download_medicare_data("2022")
    download_medicare_data("2021")
    
    print("\n" + "=" * 60)
    print("✅ All data downloaded!")
    print(f"Raw data: {DATA_RAW}")
    print(f"Labels: {DATA_LABELS}")
    print("=" * 60)
    
    print("\nNext step: Run `python scripts/filter_skin_substitutes.py`")


if __name__ == "__main__":
    main()
