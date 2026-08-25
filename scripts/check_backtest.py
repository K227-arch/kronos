import requests
r = requests.get("http://127.0.0.1:7070/api/backtest/status")
d = r.json()
print(f"Status: {d['status']}")
print(f"Progress: {d['progress']}% (Step {d['current_step']}/{d['total_steps']})")
print(f"Date: {d['current_date']}")
print(f"Stats: {d['stats']}")
print(f"Signals: {len(d['signals'])} | Trades: {len(d['trades'])}")
if d['signals']:
    print(f"Last signal: {d['signals'][-1]}")
if d['positions']:
    print(f"Open positions: {d['positions']}")
if d['error']:
    print(f"ERROR: {d['error']}")
