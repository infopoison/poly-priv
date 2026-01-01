#!/usr/bin/env python3
"""
Minimal WebSocket Connection Test
=================================

Quick validation that websocket connectivity works before running full probe.
Uses a single known-active market to test the connection.

Usage:
    python ws_connection_test.py
    
Expected: Should receive book events within 5-10 seconds.
"""

import asyncio
import json
import websockets
import requests
import sys

WS_URI = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
TEST_DURATION = 30  # seconds


def fetch_active_token():
    """Fetch a single active token from Gamma API (synchronous)"""
    print("Fetching an active market token...")
    resp = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true", "closed": "false", "limit": 10},
        timeout=10
    )
    resp.raise_for_status()
    
    markets = resp.json()
    for m in markets:
        clob_tokens = m.get("clobTokenIds")
        if clob_tokens:
            try:
                if isinstance(clob_tokens, str):
                    clob_tokens = json.loads(clob_tokens)
                token = clob_tokens[0]
                print(f"  Using token from: {m.get('question', 'Unknown')[:60]}")
                return token
            except:
                continue
    
    raise ValueError("No active tokens found")


async def test_connection():
    """Test websocket connection with a single market"""
    
    # Get an active token (synchronous)
    try:
        token_id = fetch_active_token()
    except Exception as e:
        print(f"❌ Failed to fetch active token: {e}")
        sys.exit(1)
    
    print(f"\nConnecting to {WS_URI}")
    print(f"Test duration: {TEST_DURATION}s")
    print("-" * 50)
    
    message_count = 0
    event_types = {}
    
    try:
        async with websockets.connect(WS_URI) as ws:
            print("[✓] Connected")
            
            # Subscribe
            await ws.send(json.dumps({
                "assets_ids": [token_id],
                "type": "market"
            }))
            print("[✓] Subscribed")
            
            # Start ping task
            async def ping():
                while True:
                    try:
                        await ws.send("PING")
                        await asyncio.sleep(10)
                    except:
                        break
            
            ping_task = asyncio.create_task(ping())
            
            # Receive messages
            try:
                start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start < TEST_DURATION:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        
                        if msg == "PONG":
                            continue
                        
                        message_count += 1
                        
                        try:
                            data = json.loads(msg)
                            
                            # Handle both list and dict responses
                            events = data if isinstance(data, list) else [data]
                            
                            for event in events:
                                if isinstance(event, dict):
                                    event_type = event.get("event_type", "unknown")
                                else:
                                    event_type = "non_dict"
                                event_types[event_type] = event_types.get(event_type, 0) + 1
                            
                            if message_count <= 3:
                                print(f"  [{message_count}] {len(events)} event(s): {list(set(e.get('event_type','?') for e in events if isinstance(e,dict)))[:3]}")
                            elif message_count == 4:
                                print(f"  ... (suppressing further output)")
                                
                        except json.JSONDecodeError:
                            event_types["parse_error"] = event_types.get("parse_error", 0) + 1
                            
                    except asyncio.TimeoutError:
                        print("  (waiting for messages...)")
                        
            finally:
                ping_task.cancel()
                
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        sys.exit(1)
    
    # Report
    print("-" * 50)
    print(f"\n{'='*50}")
    print(f"CONNECTION TEST RESULTS")
    print(f"{'='*50}")
    print(f"Total messages:  {message_count}")
    print(f"Message rate:    {message_count/TEST_DURATION:.2f}/sec")
    print(f"\nEvent types:")
    for et, count in sorted(event_types.items(), key=lambda x: -x[1]):
        print(f"  {et}: {count}")
    
    if message_count > 0:
        print(f"\n✅ SUCCESS - WebSocket connection working!")
        print(f"   Ready to run full Strategy B probe.")
    else:
        print(f"\n⚠️  No messages received. Market may be inactive.")
        print(f"   Try running full probe with more markets.")


if __name__ == "__main__":
    print("=" * 50)
    print("WEBSOCKET CONNECTION TEST")
    print("=" * 50)
    asyncio.run(test_connection())