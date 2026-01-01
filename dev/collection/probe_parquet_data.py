#!/usr/bin/env python3
"""
Polymarket Data Probe - Format & Schema Diagnostic
Focus: Reporting raw file structure, data types, and enrichment status.
"""

import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "../../"
BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
SAMPLE_SIZE = 5

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def probe_data():
    log("Starting Polymarket Data Format Probe...")
    
    # 1. Directory and File Check
    if not os.path.exists(BATCH_DIR):
        log(f"ERROR: Directory not found at {os.path.abspath(BATCH_DIR)}")
        return

    parquet_files = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.parquet")))
    if not parquet_files:
        log(f"ERROR: No batch files found in {BATCH_DIR}")
        return

    log(f"Total Batches Found: {len(parquet_files):,}")

    # 2. Inspect File Structure (Reading 1st file for schema)
    sample_file = parquet_files[0]
    df_sample = pd.read_parquet(sample_file)
    
    print("\n" + "="*70)
    print(f"📄 FILE SCHEMA: {os.path.basename(sample_file)}")
    print("="*70)
    
    # Report Column Types and Non-Null Counts
    schema_info = []
    for col in df_sample.columns:
        dtype = df_sample[col].dtype
        null_count = df_sample[col].isna().sum()
        sample_val = df_sample[col].iloc[0] if len(df_sample) > 0 else "N/A"
        schema_info.append({
            "Column": col,
            "Dtype": dtype,
            "Nulls": null_count,
            "Sample Value": sample_val
        })
    
    print(pd.DataFrame(schema_info).to_string(index=False))

    # 3. Aggregate Field Distributions (Sampled across files)
    all_samples = []
    indices = np.random.choice(len(parquet_files), min(SAMPLE_SIZE, len(parquet_files)), replace=False)
    for idx in indices:
        all_samples.append(pd.read_parquet(parquet_files[idx]))
    
    combined = pd.concat(all_samples, ignore_index=True)

    print("\n" + "="*70)
    print("📊 FIELD VALUE DISTRIBUTIONS (Sampled Rows)")
    print("="*70)

    # Descriptive categories
    categorical_cols = ['side', 'token_outcome', 'token_winner']
    for col in categorical_cols:
        if col in combined.columns:
            print(f"\n[{col}] unique values:")
            counts = combined[col].value_counts(dropna=False)
            for val, count in counts.items():
                print(f"  - {str(val):<15}: {count:>8,}")

    # Numeric range checks
    numeric_cols = ['price', 'maker_amount', 'taker_amount', 'size_tokens']
    print("\n[Numeric Ranges]:")
    for col in numeric_cols:
        if col in combined.columns:
            c_min = combined[col].min()
            c_max = combined[col].max()
            print(f"  - {col:<15}: min={c_min:<10} max={c_max:<10}")

    # 4. Timestamp Validation
    if 'timestamp' in combined.columns:
        raw_ts = combined['timestamp'].iloc[0]
        # Attempt conversion to verify if standard epoch or string
        try:
            ts_num = pd.to_numeric(raw_ts)
            unit = 'ms' if ts_num > 3e11 else 's'
            dt = pd.to_datetime(ts_num, unit=unit)
            print(f"\n[Timestamp Check]:")
            print(f"  - Raw Value:      {raw_ts}")
            print(f"  - Inferred Unit:  {unit}")
            print(f"  - Resolved UTC:   {dt}")
        except Exception as e:
            print(f"\n[Timestamp Error]: Could not parse '{raw_ts}' - {e}")

    print("="*70 + "\n")

if __name__ == "__main__":
    probe_data()