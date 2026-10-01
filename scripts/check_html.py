content = open(r'C:\Users\HP OMEN\Desktop\projects\trade\Kronos\webui\templates\index.html', encoding='utf-8').read()

checks = [
    ('setupEventListeners', 'function setupEventListeners'),
    ('loadAvailableModels', 'function loadAvailableModels'),
    ('updateMT5UI', 'function updateMT5UI'),
    ('setOptimalParams', 'function setOptimalParams'),
    ('runIntegratedPredict', 'function runIntegratedPredict'),
    ('autoSetupAndPredict', 'function autoSetupAndPredict'),
    ('DOMContentLoaded', 'DOMContentLoaded'),
    ('initializeApp', 'function initializeApp'),
    ('displayPredictionResult', 'function displayPredictionResult'),
]

print(f"Total lines: {content.count(chr(10))}")
for name, search in checks:
    count = content.count(search)
    status = 'OK' if count == 1 else f'FOUND {count}x' if count > 1 else 'MISSING ❌'
    print(f"  {name}: {status}")

# Count script tags
print(f"\n  <script> tags: {content.count('<script')}")
print(f"  </script> tags: {content.count('</script>')}")

# Find any syntax that might crash - unclosed braces in script
import re
script_match = re.search(r'<script>(.*)</script>', content, re.DOTALL)
if script_match:
    js = script_match.group(1)
    opens = js.count('{')
    closes = js.count('}')
    print(f"\n  JS brace balance: opens={opens} closes={closes} diff={opens-closes}")
