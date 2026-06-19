import json, time, urllib.request, re, numpy as np

KEY="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";URL="https://integrate.api.nvidia.com/v1/chat/completions";MODEL="meta/llama-3.1-70b-instruct"
FUNCS=[("merge","Combines dicts, d2 overrides d1."),("filter","Keeps items>0, list comprehension."),("flatten","Flattens list-of-lists one level."),("group","Groups dicts by key, returns dict of lists."),("normalize","(x-mean)/std, zero mean unit variance."),("tokenize","Lowercase split on whitespace."),("camel_snake","Insert _ before uppercase, lower all."),("truncate","s[:n]+... if len>n else s."),("bsearch","lo,hi,mid. Return idx or -1."),("topk","sorted+reverse+k. k highest."),("dedupe","seen set, order preserving."),("chunk","[seq[i:i+n] slice pattern]"),("read_json","json.load file into object."),("write_csv","csv.writer.writerows, no return."),("safe_rm","exists check then remove, bool."),("shuffle","Copy then shuffle, non-destructive."),("pick","random.sample no replacement."),("freq","Counter counts occurrences."),("merge_sort","Two-pointer merge with remainders."),("lru_get","LRU move-to-end hit, None on miss.")]

def call(prompt,mt=80):
    d=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0}).encode()
    for a in range(3):
        try:
            req=urllib.request.Request(URL,data=d,headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            with urllib.request.urlopen(req,timeout=30) as r:r2=json.loads(r.read())
            return r2["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e):time.sleep((a+1)*5)
    return "ERROR"

def game(null=False):
    idxs=np.random.choice(len(FUNCS),4,replace=False);tidx=np.random.randint(0,4)
    # Sender
    n,d=FUNCS[idxs[tidx]]
    sp=f"Encode into 16 floats. FUNC:{n}. DESC:{d}. Scheme:dims0-3=category(+1=data,-1=string,+1=math,-1=io),dims4-7=input count,dims8-11=output(+1=dict,-1=list),dims12-15=complexity(-1=simple,+1=complex). Output ONLY JSON array of 16 floats. JSON:"
    sr=call(sp,80)
    if null:
        z=list(np.round(np.random.randn(16)*0.5,4))
    else:
        try:m=re.search(r'\[[\d\s,.\-]+\]',sr);z=json.loads(m.group()) if m else[0]*16
        except:z=[0]*16

    # Receiver
    cs="\n".join([f"  {i+1}. {FUNCS[idx][0]} - {FUNCS[idx][1]}" for i,idx in enumerate(idxs)])
    vs=json.dumps([round(x,4) for x in z])
    rp=f"Identify which function this 16D vector encodes.\nVECTOR:{vs}\nCANDIDATES:\n{cs}\nReply ONLY number 1-4. Answer:"
    rr=call(rp,10)
    try:nums=re.findall(r'\b([1-4])\b',rr);pred=int(nums[0])-1 if nums else -1
    except:pred=-1
    return pred==tidx

if __name__=="__main__":
    print("NVIDIA LLM REFERENTIAL GAME (5 games)")
    c=nc=0
    for i in range(5):
        ok=game();nok=game(null=True);c+=ok;nc+=nok
        print(f"  Game {i+1}: signal={'OK' if ok else 'NO'} null={'OK' if nok else 'NO'}",flush=True)
    print(f"\nSignal: {c}/5={c/5:.0%}  Null: {nc}/5={nc/5:.0%}  Chance:25%")
    print(f"Verdict: {'SIGNAL' if c>=3 else 'NO SIGNAL'}")
