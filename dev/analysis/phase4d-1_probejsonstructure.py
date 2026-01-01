#!/usr/bin/env python3
"""
JSON Structure Probe - Dumps the actual structure of the phase4d JSON file
"""

import json
import sys
import argparse
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def probe_structure(obj, path="root", depth=0, max_depth=6, sample_limit=3):
    """Recursively probe and print structure."""
    indent = "  " * depth
    
    if depth > max_depth:
        print(f"{indent}... (max depth reached)")
        return
    
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}{path}: dict with {len(keys)} keys")
        
        # Show all keys if few, otherwise sample
        if len(keys) <= 10:
            keys_to_show = keys
        else:
            keys_to_show = keys[:sample_limit] + ['...'] + keys[-1:]
            print(f"{indent}  (showing {sample_limit} of {len(keys)} keys)")
        
        for k in keys_to_show:
            if k == '...':
                print(f"{indent}  ...")
                continue
            v = obj[k]
            probe_structure(v, f"['{k}']", depth + 1, max_depth, sample_limit)
    
    elif isinstance(obj, list):
        print(f"{indent}{path}: list with {len(obj)} items")
        if len(obj) > 0:
            # Show first item structure
            probe_structure(obj[0], f"[0]", depth + 1, max_depth, sample_limit)
            if len(obj) > 1:
                print(f"{indent}  ... ({len(obj)-1} more items)")
    
    else:
        # Scalar value
        type_name = type(obj).__name__
        if isinstance(obj, str) and len(obj) > 50:
            print(f"{indent}{path}: {type_name} = '{obj[:50]}...'")
        elif isinstance(obj, float):
            print(f"{indent}{path}: {type_name} = {obj:.6g}")
        else:
            print(f"{indent}{path}: {type_name} = {obj}")


def dump_surface_structure(data):
    """Specifically probe the 'surface' key which is the main data."""
    print("\n" + "="*80)
    print("DETAILED SURFACE STRUCTURE")
    print("="*80)
    
    surface = data.get('surface')
    if surface is None:
        print("No 'surface' key found!")
        print(f"Available top-level keys: {list(data.keys())}")
        return
    
    print(f"\n'surface' is a {type(surface).__name__}")
    
    if isinstance(surface, dict):
        print(f"Keys in surface: {list(surface.keys())}")
        
        # Probe first interval
        first_interval = list(surface.keys())[0] if surface else None
        if first_interval:
            print(f"\n--- Probing surface['{first_interval}'] ---")
            interval_data = surface[first_interval]
            print(f"Type: {type(interval_data).__name__}")
            
            if isinstance(interval_data, dict):
                print(f"Keys: {list(interval_data.keys())}")
                
                # Probe first threshold
                first_thresh = list(interval_data.keys())[0] if interval_data else None
                if first_thresh:
                    print(f"\n--- Probing surface['{first_interval}'][{first_thresh}] ---")
                    thresh_data = interval_data[first_thresh]
                    print(f"Type: {type(thresh_data).__name__}")
                    
                    if isinstance(thresh_data, dict):
                        print(f"Keys: {list(thresh_data.keys())}")
                        
                        # Probe first prob bucket
                        first_bucket = list(thresh_data.keys())[0] if thresh_data else None
                        if first_bucket:
                            print(f"\n--- Probing surface['{first_interval}'][{first_thresh}]['{first_bucket}'] ---")
                            bucket_data = thresh_data[first_bucket]
                            print(f"Type: {type(bucket_data).__name__}")
                            
                            if isinstance(bucket_data, dict):
                                print(f"Keys: {list(bucket_data.keys())}")
                                
                                # Show terciles structure
                                if 'terciles' in bucket_data:
                                    print(f"\n--- Probing terciles ---")
                                    terciles = bucket_data['terciles']
                                    print(f"Type: {type(terciles).__name__}")
                                    if isinstance(terciles, dict):
                                        print(f"Keys: {list(terciles.keys())}")
                                        
                                        first_terc = list(terciles.keys())[0] if terciles else None
                                        if first_terc:
                                            print(f"\n--- Full dump of terciles['{first_terc}'] ---")
                                            print(json.dumps(terciles[first_terc], indent=2, default=str))


def dump_full_sample(data):
    """Dump a complete sample path through the data."""
    print("\n" + "="*80)
    print("FULL SAMPLE CELL DUMP")
    print("="*80)
    
    surface = data.get('surface', {})
    
    # Try to find a cell with actual data
    for interval in list(surface.keys())[:3]:
        interval_data = surface[interval]
        if not isinstance(interval_data, dict):
            continue
            
        for thresh in list(interval_data.keys())[:2]:
            thresh_data = interval_data[thresh]
            if not isinstance(thresh_data, dict):
                continue
                
            for bucket in list(thresh_data.keys())[:2]:
                bucket_data = thresh_data[bucket]
                
                print(f"\n--- FULL DUMP: surface['{interval}'][{thresh}]['{bucket}'] ---")
                print(json.dumps(bucket_data, indent=2, default=str)[:3000])
                print("\n... (truncated if longer)")
                return


def check_key_types(data):
    """Check if keys are strings or other types (common issue)."""
    print("\n" + "="*80)
    print("KEY TYPE CHECK")
    print("="*80)
    
    surface = data.get('surface', {})
    
    if surface:
        first_interval = list(surface.keys())[0]
        print(f"Interval key type: {type(first_interval).__name__}, value: {repr(first_interval)}")
        
        interval_data = surface[first_interval]
        if isinstance(interval_data, dict) and interval_data:
            first_thresh = list(interval_data.keys())[0]
            print(f"Threshold key type: {type(first_thresh).__name__}, value: {repr(first_thresh)}")
            
            thresh_data = interval_data[first_thresh]
            if isinstance(thresh_data, dict) and thresh_data:
                first_bucket = list(thresh_data.keys())[0]
                print(f"Bucket key type: {type(first_bucket).__name__}, value: {repr(first_bucket)}")


def main():
    parser = argparse.ArgumentParser(description='Probe JSON structure')
    parser.add_argument('json_file', help='Path to phase4d JSON file')
    parser.add_argument('--full', action='store_true', help='Show full structure probe')
    
    args = parser.parse_args()
    
    print(f"Loading: {args.json_file}")
    data = load_json(args.json_file)
    
    # Top level overview
    print("\n" + "="*80)
    print("TOP-LEVEL STRUCTURE")
    print("="*80)
    print(f"\nTop-level keys: {list(data.keys())}")
    
    for key in data.keys():
        val = data[key]
        if isinstance(val, dict):
            print(f"  {key}: dict with {len(val)} keys")
        elif isinstance(val, list):
            print(f"  {key}: list with {len(val)} items")
        elif isinstance(val, str):
            print(f"  {key}: str = '{val[:50]}{'...' if len(val) > 50 else ''}'")
        else:
            print(f"  {key}: {type(val).__name__} = {val}")
    
    # Key types check
    check_key_types(data)
    
    # Detailed surface probe
    dump_surface_structure(data)
    
    # Full sample
    dump_full_sample(data)
    
    if args.full:
        print("\n" + "="*80)
        print("FULL RECURSIVE PROBE")
        print("="*80)
        probe_structure(data, "data", max_depth=8)


if __name__ == "__main__":
    main()