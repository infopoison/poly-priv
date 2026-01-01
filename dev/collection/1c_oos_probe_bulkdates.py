#!/usr/bin/env python3
"""
Probe: Find bulk source for market end dates
"""

import requests

ACTIVITY_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/activity-subgraph/0.0.4/gn"
GAMMA_API = "https://gamma-api.polymarket.com/markets"

def probe():
    print("="*70)
    print("PROBE: Finding bulk source for market end dates")
    print("="*70)
    
    # Test 1: What entities exist in Activity subgraph?
    print("\n" + "-"*70)
    print("TEST 1: Activity subgraph schema - what entities exist?")
    print("-"*70)
    
    query1 = """
    query {
      __schema {
        queryType {
          fields {
            name
          }
        }
      }
    }
    """
    
    resp = requests.post(ACTIVITY_SUBGRAPH, json={"query": query1}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        fields = data.get("data", {}).get("__schema", {}).get("queryType", {}).get("fields", [])
        entity_names = [f['name'] for f in fields if not f['name'].startswith('_')]
        print(f"Entities: {entity_names}")
    
    # Test 2: Check if there's a 'conditions' or 'markets' entity
    print("\n" + "-"*70)
    print("TEST 2: Check for condition/market entities with timestamps")
    print("-"*70)
    
    for entity in ['conditions', 'markets', 'resolutions', 'conditionResolutions']:
        query = f"""
        query {{
          {entity}(first: 1) {{
            id
          }}
        }}
        """
        resp = requests.post(ACTIVITY_SUBGRAPH, json={"query": query}, timeout=30)
        data = resp.json()
        if "errors" not in data:
            print(f"  ✓ '{entity}' EXISTS")
        else:
            print(f"  ✗ '{entity}' not found")
    
    # Test 3: Check Gamma API bulk endpoint
    print("\n" + "-"*70)
    print("TEST 3: Gamma API - can we query multiple conditions at once?")
    print("-"*70)
    
    # Try with multiple condition_ids
    test_conditions = [
        "0xc8134141e5dae16760a47239a81e5393fc45f8",
        "0x5c17e31f8944dc2973ad50ce033140cdaf89e8"
    ]
    
    # Try comma-separated
    resp = requests.get(GAMMA_API, params={'condition_ids': ','.join(test_conditions)}, timeout=30)
    print(f"  Comma-separated: status={resp.status_code}, results={len(resp.json()) if resp.status_code == 200 else 'N/A'}")
    
    # Try with limit/offset for bulk fetch
    resp = requests.get(GAMMA_API, params={'limit': 5, 'closed': 'true'}, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Bulk closed markets: got {len(data)} results")
        if data:
            sample = data[0]
            print(f"  Sample fields: {list(sample.keys())[:10]}")
            if 'end_date_iso' in sample:
                print(f"  ✓ end_date_iso available: {sample.get('end_date_iso')}")
            if 'condition_id' in sample:
                print(f"  ✓ condition_id available: {sample.get('condition_id')[:30]}...")
    
    # Test 4: Can Gamma paginate through all closed markets?
    print("\n" + "-"*70)
    print("TEST 4: Gamma API pagination for closed markets")
    print("-"*70)
    
    resp = requests.get(GAMMA_API, params={'limit': 100, 'closed': 'true', 'offset': 0}, timeout=30)
    if resp.status_code == 200:
        batch1 = resp.json()
        print(f"  Offset 0: {len(batch1)} markets")
        
        resp = requests.get(GAMMA_API, params={'limit': 100, 'closed': 'true', 'offset': 100}, timeout=30)
        batch2 = resp.json()
        print(f"  Offset 100: {len(batch2)} markets")
        
        resp = requests.get(GAMMA_API, params={'limit': 100, 'closed': 'true', 'offset': 10000}, timeout=30)
        batch3 = resp.json()
        print(f"  Offset 10000: {len(batch3)} markets")
        
        if len(batch1) == 100 and len(batch2) == 100:
            print(f"\n  ✓ Gamma API supports pagination!")
            print(f"  We can bulk fetch all closed markets with end_date_iso")
    
    # Test 5: Check if Gamma has date filtering
    print("\n" + "-"*70)
    print("TEST 5: Gamma API date filtering")
    print("-"*70)
    
    # Try various date filter params
    for param in ['end_date_min', 'end_date_gt', 'start_date_min', 'created_after']:
        resp = requests.get(GAMMA_API, params={'limit': 5, param: '2025-12-08'}, timeout=30)
        if resp.status_code == 200 and len(resp.json()) > 0:
            print(f"  ✓ '{param}' works: got {len(resp.json())} results")
        else:
            print(f"  ✗ '{param}' doesn't work or no results")

    print("\n" + "="*70)
    print("PROBE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    probe()