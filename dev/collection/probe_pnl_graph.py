#!/usr/bin/env python3
"""
Probe: Can we get resolved conditions directly from PNL subgraph?
Skip the redemptions entirely.
"""

import requests
from datetime import datetime, timezone

PNL_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn"

def probe():
    print("="*70)
    print("PNL SUBGRAPH PROBE - Looking for resolved conditions directly")
    print("="*70)
    
    # Test 1: What fields does a condition have?
    print("\n" + "-"*70)
    print("TEST 1: Introspect condition schema")
    print("-"*70)
    
    query1 = """
    query {
      __type(name: "Condition") {
        fields {
          name
          type {
            name
            kind
          }
        }
      }
    }
    """
    
    resp = requests.post(PNL_SUBGRAPH, json={"query": query1}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        fields = data.get("data", {}).get("__type", {}).get("fields", [])
        print(f"Condition has {len(fields)} fields:")
        for f in fields:
            print(f"  - {f['name']}: {f['type']}")
    
    # Test 2: Get first 5 conditions with payouts
    print("\n" + "-"*70)
    print("TEST 2: First 5 conditions (any)")
    print("-"*70)
    
    query2 = """
    query {
      conditions(first: 5, orderBy: id, orderDirection: asc) {
        id
        payoutNumerators
        payoutDenominator
      }
    }
    """
    
    resp = requests.post(PNL_SUBGRAPH, json={"query": query2}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        conditions = data.get("data", {}).get("conditions", [])
        print(f"Returned: {len(conditions)} conditions")
        for c in conditions:
            print(f"  id={c['id'][:30]}... payout={c.get('payoutNumerators')} / {c.get('payoutDenominator')}")
    
    # Test 3: Get conditions where payoutNumerators is not empty (resolved)
    print("\n" + "-"*70)
    print("TEST 3: Conditions with payoutNumerators (resolved markets)")
    print("-"*70)
    
    query3 = """
    query {
      conditions(
        first: 10, 
        orderBy: id, 
        orderDirection: desc,
        where: { payoutNumerators_not: [] }
      ) {
        id
        payoutNumerators
        payoutDenominator
      }
    }
    """
    
    print(f"Query: where: {{ payoutNumerators_not: [] }}")
    
    resp = requests.post(PNL_SUBGRAPH, json={"query": query3}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        conditions = data.get("data", {}).get("conditions", [])
        print(f"Returned: {len(conditions)} conditions")
        for c in conditions:
            print(f"  id={c['id'][:30]}... payout={c.get('payoutNumerators')}")
    
    # Test 4: Check if conditions have a resolutionTimestamp field
    print("\n" + "-"*70)
    print("TEST 4: Check for timestamp/resolution fields on conditions")
    print("-"*70)
    
    query4 = """
    query {
      conditions(first: 1) {
        id
        payoutNumerators
        payoutDenominator
        resolutionTimestamp
        resolvedAt
        timestamp
        createdAt
      }
    }
    """
    
    resp = requests.post(PNL_SUBGRAPH, json={"query": query4}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"Errors (expected if fields don't exist): {data['errors']}")
    else:
        conditions = data.get("data", {}).get("conditions", [])
        if conditions:
            print(f"Condition fields: {conditions[0]}")
    
    # Test 5: Try alternative resolution field names
    print("\n" + "-"*70)
    print("TEST 5: Try resolutionTimestamp field")
    print("-"*70)
    
    query5 = """
    query {
      conditions(first: 5, where: { payoutNumerators_not: [] }) {
        id
        payoutNumerators
      }
    }
    """
    
    resp = requests.post(PNL_SUBGRAPH, json={"query": query5}, timeout=30)
    data = resp.json()
    
    if "errors" in data:
        print(f"ERROR: {data['errors']}")
    else:
        conditions = data.get("data", {}).get("conditions", [])
        print(f"Got {len(conditions)} resolved conditions")
        for c in conditions:
            print(f"  {c['id'][:40]}... -> {c['payoutNumerators']}")

    # Test 6: Count resolved conditions (paginate a bit)
    print("\n" + "-"*70)
    print("TEST 6: Count resolved conditions (sample)")
    print("-"*70)
    
    total = 0
    last_id = ""
    batches = 0
    
    while batches < 5:  # Just 5 batches to estimate
        query6 = f"""
        query {{
          conditions(
            first: 1000, 
            orderBy: id, 
            orderDirection: asc,
            where: {{ 
              payoutNumerators_not: [],
              id_gt: "{last_id}"
            }}
          ) {{
            id
          }}
        }}
        """
        
        resp = requests.post(PNL_SUBGRAPH, json={"query": query6}, timeout=30)
        data = resp.json()
        
        if "errors" in data:
            print(f"ERROR: {data['errors']}")
            break
        
        conditions = data.get("data", {}).get("conditions", [])
        if not conditions:
            break
            
        total += len(conditions)
        last_id = conditions[-1]['id']
        batches += 1
        print(f"  Batch {batches}: {len(conditions)} conditions (total so far: {total})")
        
        if len(conditions) < 1000:
            print(f"  (Last batch - total resolved conditions: {total})")
            break
    
    if batches == 5:
        print(f"  (Stopped at 5 batches - at least {total} resolved conditions exist)")

    print("\n" + "="*70)
    print("PROBE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    probe()