import pandas as pd
import pyarrow.parquet as pq
import glob
import os
import json
import requests

# Set your absolute path to avoid the directory disappearance issue
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../")) 
BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
MARKETS_CSV = os.path.join(BASE_DIR, 'markets_past_year.csv')

def get_market_metadata(condition_id):
    """Fetch raw API metadata to see the clobTokenIds array."""
    try:
        url = f"https://gamma-api.polymarket.com/markets?condition_ids={condition_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data[0] if isinstance(data, list) and len(data) > 0 else None
    except: return None
    return None

def run_deep_diagnostic(num_samples=15):
    print(f"{'Condition':<12} | {'Token ID':<10} | {'Price':<6} | {'Data_Win':<8} | {'CSV_Out'}")
    print("-" * 65)
    
    truth_df = pd.read_csv(MARKETS_CSV)
    truth_map = dict(zip(truth_df['condition_id'].astype(str), truth_df['outcome']))
    
    files = glob.glob(os.path.join(BATCH_DIR, "*.parquet"))
    found = 0

    for f in files:
        if found >= num_samples: break
        df = pq.read_table(f).to_pandas()
        
        # Target the discrepancies seen in your screenshot (high price, outcome 0)
        for _, row in df.iterrows():
            cid = str(row['condition_id'])
            csv_outcome = truth_map.get(cid)
            
            # Focus only on high-confidence discrepancies for analysis
            if row['price'] > 0.90 and csv_outcome == 0 and found < num_samples:
                token_id = str(row['token_id'])
                meta = get_market_metadata(cid)
                
                clob_ids = meta.get('clobTokenIds', '[]') if meta else "N/A"
                print(f"{cid[:10]}... | {token_id[:8]}... | {row['price']:.4f} | {row.get('token_winner'):<8} | {csv_outcome}")
                print(f"   RAW METADATA CLOB_IDS: {clob_ids}")
                
                # Verify if this token_id is at index 0 or index 1
                if meta and isinstance(clob_ids, list):
                    try:
                        idx = clob_ids.index(token_id)
                        print(f"   DIAGNOSTIC: Token is at Index {idx}. Market Truth is {csv_outcome}.")
                        if idx != csv_outcome:
                            print("   ⚠️ ERROR: This token is a LOSER but script/data marked it Winner.")
                    except ValueError:
                        print("   ⚠️ ERROR: Token ID not found in clobTokenIds array.")
                
                print("-" * 65)
                found += 1

if __name__ == "__main__":
    run_deep_diagnostic()