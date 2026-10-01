import re

content = open(r'C:\Users\HP OMEN\Desktop\projects\trade\Kronos\webui\templates\index.html', encoding='utf-8').read()
lines = content.splitlines()

# Find both occurrences of runIntegratedPredict function definition
print("=== runIntegratedPredict definitions ===")
for i, l in enumerate(lines, 1):
    if 'function runIntegratedPredict' in l:
        print(f"  Line {i}: {l.strip()[:80]}")

# Find the brace imbalance — scan JS section line by line
script_start = content.find('<script>')
script_end   = content.rfind('</script>')
js_content   = content[script_start:script_end]
js_lines     = js_content.splitlines()

opens = 0; closes = 0
for i, l in enumerate(js_lines, 1):
    o = l.count('{') - l.count('${')  # exclude template literals
    c = l.count('}')
    opens  += o
    closes += c
    bal = opens - closes
    # Report when balance goes very high or very low
    if bal > 15 or bal < 0:
        print(f"  JS line {i} [bal={bal:+d}]: {l.strip()[:80]}")

print(f"\nFinal balance: {opens-closes}")
