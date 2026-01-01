#!/usr/bin/env python3
"""
Probe: Does CLOB API have resolution timestamp?
"""

import requests
import json

CLOB_API_URL = "https://clob.polymarket.com/markets"

# A known resolved condition from the PNL probe
TEST_CONDITION = "0x0001bd6b1ce49b28d822af08b0ff1844bf789bfeb7634a88b45e7619a0d45837"

def probe():
    print("="*70)
    print("CLOB API PROBE - Looking for resolution timestamp")
    print("="*70)
    
    url = f"{CLOB_API_URL}/{TEST_CONDITION}"
    print(f"\nFetching: {url}\n")
    
    resp = requests.get(url, timeout=30)
    
    if resp.status_code != 200:
        print(f"ERROR: Status {resp.status_code}")
        print(resp.text[:500])
        return
    
    data = resp.json()
    
    print("Full response (formatted):")
    print("-"*70)
    print(json.dumps(data, indent=2))
    print("-"*70)
    
    # Look for timestamp-related fields
    print("\nLooking for timestamp fields...")
    
    def find_timestamps(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if 'time' in k.lower() or 'date' in k.lower() or 'stamp' in k.lower() or 'end' in k.lower() or 'resolv' in k.lower() or 'close' in k.lower():
                    print(f"  {prefix}{k}: {v}")
                find_timestamps(v, prefix + k + ".")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_timestamps(item, prefix + f"[{i}].")
    
    find_timestamps(data)

if __name__ == "__main__":
    probe()