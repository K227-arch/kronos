import requests

r = requests.post('http://127.0.0.1:7070/api/news/fetch', json={'query': 'Gold XAUUSD'}, timeout=15)
d = r.json()
print("Success:", d.get('success'))
print("Error:", d.get('error'))
s = d.get('sentiment', {})
print(f"Articles: {s.get('articles')}")
print(f"Signal: {s.get('signal')}")
print(f"Score: {s.get('score')}")
print(f"Bias: {s.get('bias')}%")
arts = d.get('articles', [])
for a in arts[:3]:
    print(f"\n  [{a.get('source')}] {a.get('title','')[:80]}")
    print(f"  Polarity: {a.get('polarity')} | Relevance: {a.get('relevance')}")
