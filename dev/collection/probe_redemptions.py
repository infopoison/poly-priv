#!/usr/bin/env python3
"""
Probe: Debug redemptions query timestamp filtering
"""

import requests
from datetime import datetime, timezone

ACTIVITY_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/activity-subgraph/0.0.4/gn"

# The timestamp we're trying to filter on
AFTER_TIMESTAMP = 1765158377  # Dec 8, 2025

def probe():
    print("="*70)
    print("REDEMPTIONS QUERY PROBE")
    print("="*70)
    
    print(f"\nTarget filter: timestamp >= {AFTER_TIMESTAMP}")
    print(f"Which is: {datetime.fromtimestamp(AFTER_TIMESTAMP, tz=timezone.utc)}")
    
    # Query 1: Fetch first 5 redemptions WITHOUT any filter
    print("\n" + "-"*70)
    print("TEST 1: First 5 redemptions (NO filter)")
    print("-"*70)
    
    query1 = """
    query {
      redemptions(first: 5, orderBy: timestamp, orderDirection: asc) {
        id
        timestamp
        condition
      }
    }
    """
    
    resp = requests.post(ACTIVITY_SUBGRAPH, json={"query": query1}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        redemptions = data.get("data", {}).get("redemptions", [])
        print(f"Returned: {len(redemptions)} redemptions")
        for r in redemptions:
            ts = int(r['timestamp'])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            print(f"  timestamp={ts} ({dt}) condition={r['condition'][:20]}...")
    
    # Query 2: Fetch first 5 redemptions WITH timestamp_gte filter
    print("\n" + "-"*70)
    print(f"TEST 2: First 5 redemptions WITH timestamp_gte: {AFTER_TIMESTAMP}")
    print("-"*70)
    
    query2 = f"""
    query {{
      redemptions(
        first: 5, 
        orderBy: timestamp, 
        orderDirection: asc,
        where: {{ timestamp_gte: {AFTER_TIMESTAMP} }}
      ) {{
        id
        timestamp
        condition
      }}
    }}
    """
    
    print(f"Query:\n{query2}")
    
    resp = requests.post(ACTIVITY_SUBGRAPH, json={"query": query2}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        redemptions = data.get("data", {}).get("redemptions", [])
        print(f"Returned: {len(redemptions)} redemptions")
        for r in redemptions:
            ts = int(r['timestamp'])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            print(f"  timestamp={ts} ({dt}) condition={r['condition'][:20]}...")
    
    # Query 3: Fetch last 5 redemptions (most recent)
    print("\n" + "-"*70)
    print("TEST 3: Last 5 redemptions (most recent, descending)")
    print("-"*70)
    
    query3 = """
    query {
      redemptions(first: 5, orderBy: timestamp, orderDirection: desc) {
        id
        timestamp
        condition
      }
    }
    """
    
    resp = requests.post(ACTIVITY_SUBGRAPH, json={"query": query3}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        redemptions = data.get("data", {}).get("redemptions", [])
        print(f"Returned: {len(redemptions)} redemptions")
        for r in redemptions:
            ts = int(r['timestamp'])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            print(f"  timestamp={ts} ({dt}) condition={r['condition'][:20]}...")
    
    # Query 4: Count total redemptions (approximate)
    print("\n" + "-"*70)
    print("TEST 4: Attempting to get total count")
    print("-"*70)
    
    query4 = """
    query {
      redemptions(first: 1000, orderBy: id, orderDirection: desc) {
        id
      }
    }
    """
    
    resp = requests.post(ACTIVITY_SUBGRAPH, json={"query": query4}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        redemptions = data.get("data", {}).get("redemptions", [])
        print(f"Returned: {len(redemptions)} redemptions in sample")
        if redemptions:
            print(f"Highest ID in sample: {redemptions[0]['id']}")
    
    # Query 5: Test with string timestamp (in case that's the issue)
    print("\n" + "-"*70)
    print(f"TEST 5: timestamp_gte as STRING: \"{AFTER_TIMESTAMP}\"")
    print("-"*70)
    
    query5 = f"""
    query {{
      redemptions(
        first: 5, 
        orderBy: timestamp, 
        orderDirection: asc,
        where: {{ timestamp_gte: "{AFTER_TIMESTAMP}" }}
      ) {{
        id
        timestamp
        condition
      }}
    }}
    """
    
    print(f"Query:\n{query5}")
    
    resp = requests.post(ACTIVITY_SUBGRAPH, json={"query": query5}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        redemptions = data.get("data", {}).get("redemptions", [])
        print(f"Returned: {len(redemptions)} redemptions")
        for r in redemptions:
            ts = int(r['timestamp'])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            print(f"  timestamp={ts} ({dt}) condition={r['condition'][:20]}...")

    print("\n" + "="*70)
    print("PROBE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    probe()