#!/usr/bin/env python3
"""
OOS Market Discovery v3 - Integrated Flow with Optional Mapping Cache
=====================================================================

Uses redemptions (time-filtered) + deduplication approach.
Mapping file is OPTIONAL - will fetch from Positions subgraph if not provided.

Usage:
    # Fresh run (fetches mapping from Positions subgraph):
    python oos_discover_markets_v3.py --after-date 2025-12-08 --output-dir oos_collection/
    
    # With existing mapping (skips Stage A):
    python oos_discover_markets_v3.py --after-date 2025-12-08 --output-dir oos_collection/ \
        --mapping-file oos_collection/oos_token_mapping.csv
"""

import argparse
import csv
import json
import os
import sys
import time
import requests
from datetime import datetime, timezone
from collections import defaultdict

# ==========================================
# API ENDPOINTS
# ==========================================
POSITIONS_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/positions-subgraph/0.0.7/gn"
ACTIVITY_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/activity-subgraph/0.0.4/gn"
PNL_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn"

REQUEST_DELAY = 0.5
BATCH_SIZE = 1000

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

def log_stage(stage_name):
    print("\n" + "="*70)
    print(f"  {stage_name}")
    print("="*70 + "\n")

def execute_graphql(url, query, description="query", max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                log(f"GraphQL error in {description}: {data['errors']}", "ERROR")
                return None
            
            return data.get("data")
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 10s, 20s, 30s, 40s
                log(f"HTTP error in {description}: {e}. Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})", "WARN")
                time.sleep(wait_time)
            else:
                log(f"HTTP error in {description} after {max_retries} attempts: {e}", "ERROR")
                return None
    
    return None

# ==========================================
# STAGE A: TOKEN MAPPING (Optional - from file or Positions subgraph)
# ==========================================
def load_token_mapping_from_file(mapping_file):
    """Load existing token mapping CSV"""
    log(f"Loading mapping from {mapping_file}...")
    
    condition_to_tokens = defaultdict(list)
    row_count = 0
    
    with open(mapping_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_id = row['token_id']
            condition_id = row['condition_id']
            condition_to_tokens[condition_id].append(token_id)
            row_count += 1
            
            if row_count % 100000 == 0:
                log(f"  Read {row_count:,} rows...")
    
    log(f"✓ Loaded {row_count:,} token mappings")
    log(f"✓ {len(condition_to_tokens):,} unique conditions")
    
    return condition_to_tokens

def fetch_token_mapping_from_subgraph(output_dir):
    """Fetch token mapping from Positions subgraph"""
    log("Fetching from Positions subgraph...")
    log("(This may take 10-20 minutes for ~1M+ mappings)\n")
    
    output_file = os.path.join(output_dir, "oos_token_mapping.csv")
    
    all_mappings = []
    last_id = "0"
    batch_num = 0
    start_time = time.time()
    
    while True:
        batch_num += 1
        
        query = f"""
        query GetTokenIdConditions {{
          tokenIdConditions(
            first: {BATCH_SIZE},
            orderBy: id,
            orderDirection: asc,
            where: {{ id_gt: "{last_id}" }}
          ) {{
            id
            condition {{
              id
            }}
          }}
        }}
        """
        
        data = execute_graphql(POSITIONS_SUBGRAPH, query, f"token mapping batch {batch_num}")
        
        if data is None:
            log("Failed to fetch token mappings", "ERROR")
            return None
        
        batch = data.get("tokenIdConditions", [])
        
        if len(batch) == 0:
            break
        
        all_mappings.extend(batch)
        last_id = batch[-1]["id"]
        
        if batch_num % 100 == 0:
            elapsed = time.time() - start_time
            rate = len(all_mappings) / elapsed
            log(f"  Batch {batch_num}: {len(all_mappings):,} mappings ({rate:.0f}/sec)")
        
        if len(batch) < BATCH_SIZE:
            break
        
        time.sleep(REQUEST_DELAY)
    
    elapsed = time.time() - start_time
    log(f"\n✓ Fetched {len(all_mappings):,} token mappings in {elapsed:.1f}s")
    
    # Save to CSV for future use
    log(f"Saving to {output_file} for future runs...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['token_id', 'condition_id'])
        for item in all_mappings:
            writer.writerow([item['id'], item['condition']['id']])
    
    # Build reverse mapping
    condition_to_tokens = defaultdict(list)
    for item in all_mappings:
        condition_to_tokens[item['condition']['id']].append(item['id'])
    
    log(f"✓ {len(condition_to_tokens):,} unique conditions")
    
    return condition_to_tokens

def get_token_mapping(mapping_file, output_dir):
    """Get token mapping from file or subgraph"""
    log_stage("STAGE A: Token → Condition Mapping")
    
    if mapping_file and os.path.exists(mapping_file):
        log(f"Using existing mapping file: {mapping_file}")
        return load_token_mapping_from_file(mapping_file)
    elif mapping_file:
        log(f"Mapping file not found: {mapping_file}", "WARN")
        log("Falling back to Positions subgraph...")
        return fetch_token_mapping_from_subgraph(output_dir)
    else:
        log("No mapping file provided, fetching from Positions subgraph...")
        return fetch_token_mapping_from_subgraph(output_dir)

# ==========================================
# STAGE B: RESOLVED MARKETS (Redemptions + Dedupe)
# ==========================================
def fetch_resolved_markets(after_timestamp, before_timestamp, output_dir):
    """Fetch markets resolved in time window via redemptions"""
    log_stage("STAGE B: Fetching Resolved Markets")
    
    output_file = os.path.join(output_dir, "oos_resolved_markets.csv")
    checkpoint_file = os.path.join(output_dir, "oos_redemptions_checkpoint.json")
    
    log(f"Time window:")
    log(f"  After:  {datetime.fromtimestamp(after_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if before_timestamp:
        log(f"  Before: {datetime.fromtimestamp(before_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        log(f"  Before: (now)")
    
    # Step 1+2: Fetch redemptions AND deduplicate on-the-fly
    # Only keep condition_times dict in memory, NOT all redemptions
    log("\nStep 1/2: Fetching redemptions + deduplicating on-the-fly...")
    log("(Only unique conditions kept in memory, not all redemption events)\n")
    
    # Load checkpoint if exists
    condition_times = {}
    last_id = ""
    batch_num = 0
    total_redemptions = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                ckpt = json.load(f)
            condition_times = ckpt.get('condition_times', {})
            last_id = ckpt.get('last_id', "")
            batch_num = ckpt.get('batch_num', 0)
            total_redemptions = ckpt.get('total_redemptions', 0)
            log(f"✓ Resuming from checkpoint: batch {batch_num}, {total_redemptions:,} redemptions, {len(condition_times):,} unique")
        except Exception as e:
            log(f"Failed to load checkpoint: {e}, starting fresh", "WARN")
    
    start_time = time.time()
    
    while True:
        batch_num += 1
        
        where_parts = [f'timestamp_gte: {after_timestamp}']
        if before_timestamp:
            where_parts.append(f'timestamp_lte: {before_timestamp}')
        if last_id:
            where_parts.append(f'id_gt: "{last_id}"')
        
        where_clause = ", ".join(where_parts)
        
        query = f"""
        query FetchRedemptions {{
          redemptions(
            first: {BATCH_SIZE},
            orderBy: id,
            orderDirection: asc,
            where: {{ {where_clause} }}
          ) {{
            id
            timestamp
            condition
          }}
        }}
        """
        
        data = execute_graphql(ACTIVITY_SUBGRAPH, query, f"redemptions batch {batch_num}")
        
        if data is None:
            log(f"Failed to fetch batch {batch_num}, saving checkpoint and stopping", "ERROR")
            # Save checkpoint before failing
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'condition_times': condition_times,
                    'last_id': last_id,
                    'batch_num': batch_num - 1,
                    'total_redemptions': total_redemptions
                }, f)
            log(f"Checkpoint saved. Restart to resume from batch {batch_num}", "INFO")
            return None
        
        batch = data.get("redemptions", [])
        
        if len(batch) == 0:
            break
        
        # Deduplicate on-the-fly - only keep earliest timestamp per condition
        for redemption in batch:
            cond_id = redemption['condition']
            ts = int(redemption['timestamp'])
            
            if cond_id not in condition_times or ts < condition_times[cond_id]:
                condition_times[cond_id] = ts
        
        total_redemptions += len(batch)
        last_id = batch[-1]['id']
        
        # Clear batch immediately
        del batch
        
        if batch_num % 100 == 0:
            elapsed = time.time() - start_time
            rate = total_redemptions / elapsed if elapsed > 0 else 0
            log(f"  Batch {batch_num}: {total_redemptions:,} redemptions → {len(condition_times):,} unique ({rate:.0f}/sec)")
            
            # Save checkpoint every 500 batches
            if batch_num % 500 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'condition_times': condition_times,
                        'last_id': last_id,
                        'batch_num': batch_num,
                        'total_redemptions': total_redemptions
                    }, f)
        
        if len(data.get("redemptions", [])) < BATCH_SIZE:
            break
        
        time.sleep(REQUEST_DELAY)
    
    elapsed = time.time() - start_time
    log(f"\n✓ Processed {total_redemptions:,} redemptions → {len(condition_times):,} unique conditions in {elapsed:.1f}s")
    
    # Clear checkpoint file since we're done with redemptions
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Step 3: Fetch outcomes
    log(f"\nStep 3/3: Fetching outcomes for {len(condition_times):,} conditions...")
    
    results = {}
    skipped = 0
    
    for i, (condition_id, resolution_time) in enumerate(condition_times.items()):
        if (i + 1) % 100 == 0:
            log(f"  [{i+1}/{len(condition_times)}] Fetching outcomes... ({len(results)} valid)")
        
        query = f"""
        query GetOutcome {{
          condition(id: "{condition_id}") {{
            id
            payoutNumerators
            payoutDenominator
          }}
        }}
        """
        
        data = execute_graphql(PNL_SUBGRAPH, query, "outcome query")
        time.sleep(REQUEST_DELAY)
        
        if data is None or data.get("condition") is None:
            skipped += 1
            continue
        
        payout = data["condition"]
        numerators = payout.get('payoutNumerators', [])
        
        if len(numerators) != 2:
            skipped += 1
            continue
        
        if numerators[0] == '1' and numerators[1] == '0':
            outcome = 0
        elif numerators[0] == '0' and numerators[1] == '1':
            outcome = 1
        else:
            skipped += 1
            continue
        
        results[condition_id] = {
            'resolution_time': resolution_time,
            'outcome': outcome
        }
    
    log(f"\n✓ {len(results):,} markets with valid outcomes")
    log(f"  Skipped: {skipped} (non-binary or missing)")
    
    # Save to CSV
    log(f"\nWriting to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition_id', 'resolution_time', 'outcome'])
        writer.writeheader()
        for cond_id, data in results.items():
            writer.writerow({
                'condition_id': cond_id,
                'resolution_time': data['resolution_time'],
                'outcome': data['outcome']
            })
    
    return results

# ==========================================
# STAGE C: JOIN
# ==========================================
def join_and_output(condition_to_tokens, resolved_markets, output_dir):
    """Join token mapping with resolved markets"""
    log_stage("STAGE C: Creating Backfill Token List")
    
    output_file = os.path.join(output_dir, "oos_backfill_tokens.csv")
    
    backfill_list = []
    markets_with_tokens = 0
    markets_without_tokens = 0
    
    for condition_id, market_data in resolved_markets.items():
        tokens = condition_to_tokens.get(condition_id, [])
        
        if len(tokens) == 2:
            for token_id in tokens:
                backfill_list.append({
                    'token_id': token_id,
                    'condition_id': condition_id,
                    'resolution_time': market_data['resolution_time'],
                    'outcome': market_data['outcome']
                })
            markets_with_tokens += 1
        else:
            markets_without_tokens += 1
    
    log(f"Join results:")
    log(f"  Markets with tokens:    {markets_with_tokens:,}")
    log(f"  Markets without tokens: {markets_without_tokens:,}")
    log(f"  Total tokens for backfill: {len(backfill_list):,}")
    
    log(f"\nWriting to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['token_id', 'condition_id', 'resolution_time', 'outcome'])
        writer.writeheader()
        writer.writerows(backfill_list)
    
    return backfill_list

# ==========================================
# MANIFEST
# ==========================================
def write_manifest(output_dir, args, num_markets, num_tokens):
    manifest_file = os.path.join(output_dir, "oos_discovery_manifest.json")
    
    queries_per_token = 2
    avg_pagination = 1.5
    delay_per_query = 5.0
    total_queries = num_tokens * queries_per_token * avg_pagination
    total_hours = (total_queries * delay_per_query) / 3600
    
    manifest = {
        'created_at': datetime.now(tz=timezone.utc).isoformat(),
        'script_version': 'v3_integrated',
        'parameters': {
            'after_timestamp': args.after,
            'after_date': datetime.fromtimestamp(args.after, tz=timezone.utc).isoformat(),
            'before_timestamp': args.before,
            'mapping_file': args.mapping_file,
        },
        'results': {
            'resolved_markets': num_markets,
            'tokens_for_backfill': num_tokens,
        },
        'runtime_estimate': {
            'queries': int(total_queries),
            'hours': round(total_hours, 1),
            'days': round(total_hours / 24, 2)
        },
        'next_step': f"python oos_fetch_order_history.py --input {output_dir}/oos_backfill_tokens.csv --output-dir {output_dir}/order_history_batches/"
    }
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    log(f"✓ Manifest written to {manifest_file}")
    return manifest['runtime_estimate']

# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="OOS Market Discovery v3 (Integrated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fresh run (fetches token mapping from Positions subgraph):
    python oos_discover_markets_v3.py --after-date 2025-12-08 --output-dir oos_collection/
    
    # With existing mapping (skips Stage A):
    python oos_discover_markets_v3.py --after-date 2025-12-08 --output-dir oos_collection/ \\
        --mapping-file oos_collection/oos_token_mapping.csv
        """
    )
    
    parser.add_argument('--after', type=int, help='Unix timestamp: markets resolved AFTER this')
    parser.add_argument('--after-date', type=str, help='ISO date (YYYY-MM-DD)')
    parser.add_argument('--before', type=int, help='Unix timestamp: markets resolved BEFORE this')
    parser.add_argument('--before-date', type=str, help='ISO date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--mapping-file', type=str, default=None,
                        help='Optional: path to existing token mapping CSV (skips Stage A)')
    
    args = parser.parse_args()
    
    # Parse dates
    if args.after_date:
        args.after = int(datetime.strptime(args.after_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    if args.before_date:
        args.before = int(datetime.strptime(args.before_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    
    if not args.after:
        parser.error("Must specify --after or --after-date")
    
    # Safety
    if args.output_dir in ['order_history_batches', '.', './']:
        log("ERROR: Cannot use 'order_history_batches' as output", "ERROR")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log("="*70)
    log("  OOS MARKET DISCOVERY v3 (Integrated)")
    log("="*70)
    log(f"\nOutput directory: {os.path.abspath(args.output_dir)}")
    log(f"Mapping file:     {args.mapping_file or '(will fetch from Positions subgraph)'}")
    log(f"After:  {datetime.fromtimestamp(args.after, tz=timezone.utc).strftime('%Y-%m-%d')}")
    if args.before:
        log(f"Before: {datetime.fromtimestamp(args.before, tz=timezone.utc).strftime('%Y-%m-%d')}")
    
    # Stage A: Get token mapping
    condition_to_tokens = get_token_mapping(args.mapping_file, args.output_dir)
    if condition_to_tokens is None:
        sys.exit(1)
    
    # Stage B: Resolved markets
    resolved_markets = fetch_resolved_markets(args.after, args.before, args.output_dir)
    if resolved_markets is None:
        sys.exit(1)
    
    # Stage C: Join
    backfill_list = join_and_output(condition_to_tokens, resolved_markets, args.output_dir)
    
    # Manifest
    runtime = write_manifest(args.output_dir, args, len(resolved_markets), len(backfill_list))
    
    # Summary
    log_stage("DISCOVERY COMPLETE")
    log(f"Resolved markets: {len(resolved_markets):,}")
    log(f"Tokens for backfill: {len(backfill_list):,}")
    log(f"\nEstimated fetch time: ~{runtime['hours']} hours ({runtime['days']} days)")
    log(f"\nNext step:")
    log(f"  python oos_fetch_order_history.py --input {args.output_dir}/oos_backfill_tokens.csv --output-dir {args.output_dir}/order_history_batches/")
    log("="*70)

if __name__ == "__main__":
    main()