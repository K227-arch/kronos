import requests
r = requests.get("http://127.0.0.1:7070/api/backtest/status")
d = r.json()
print(f"Status: {d['status']} | Progress: {d['progress']}%")
print(f"Stats: {d['stats']}")
print(f"\nLast 5 trades:")
for t in d['trades'][-5:]:
    print(f"  {t}")
print(f"\nLast 3 signals:")
for s in d['signals'][-3:]:
    print(f"  {s}")
print(f"\nPositions: {d['positions']}")
