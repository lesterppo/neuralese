import json, time, urllib.request, re, numpy as np

KEY = "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"
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-70b-instruct"
FUNCS = [
    ("merge","Combines two dicts, d2 overrides d1."),
    ("filter","Keeps items>0 via comprehension."),
    ("flatten","Flattens list-of-lists one level."),
    ("group","Groups dicts by key, returns dict of lists."),
]

def call(prompt, mt=80):
    d = json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0}).encode()
    for a in range(3):
        try:
            req = urllib.request.Request(URL, data=d, headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            with urllib.request.urlopen(req, timeout=30) as r:
                r2 = json.loads(r.read())
            return r2["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e): time.sleep((a+1)*5)
    return "ERROR"

# Test sender
prompt = "Encode into 16 floats. FUNC: merge. DESC: Combines two dicts, d2 overrides d1. Scheme: dims0-3=category(+1=data),dims4-7=input count,dims8-11=output(+1=dict),dims12-15=complexity. Output ONLY JSON array of 16 floats. JSON:"
resp = call(prompt, 80)
print(f"Sender: {resp[:200]}")

# Parse
try:
    m = re.search(r'\[[\d\s,.\-]+\]', resp)
    z = json.loads(m.group()) if m else None
    print(f"Vector ({len(z) if z else 0}d): {z}")
except Exception as e:
    print(f"Parse error: {e}")
