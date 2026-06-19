import json, time, urllib.request, sys, os, re, numpy as np

# Read OpenRouter key from .env
KEY = ""
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        if line.startswith("OPENROUTER_API_KEY="):
            KEY = line.strip().split("=", 1)[1]
            break
if not KEY:
    print("No key"); sys.exit(1)

URL = "https://openrouter.ai/api/v1/chat/completions"
HD = {"Content-Type":"application/json","Authorization":"Bearer "+KEY,
      "HTTP-Referer":"https://github.com/lesterppo/neuralese"}
LATENT_DIM,N=16,4
MODELS=["google/gemma-4-26b-a4b-it:free","openai/gpt-oss-120b:free","nvidia/nemotron-3-super-120b-a12b:free"]
FUNCS=[("merge","Combines two dicts, d2 overrides d1."),("filter","Keeps items>0 via comprehension."),
("flatten","Flattens list-of-lists one level."),("group","Groups dicts by key, returns dict of lists."),
("normalize","(x-mean)/std, zero mean unit variance."),("tokenize","Lowercase split on whitespace."),
("camel_snake","Insert _ before uppercase, lower all."),("truncate","s[:n]+... if len>n else s."),
("bsearch","lo,hi,mid pattern. Return idx or -1."),("topk","sorted+reverse+k, returns k highest."),
("dedupe","seen set preserves order."),("chunk","[seq[i:i+n] slice pattern]"),
("read_json","json.load file into Python object."),("write_csv","csv.writer.writerows, no return."),
("safe_rm","exists check then remove, returns bool."),("shuffle","Copy then shuffle, non-destructive."),
("pick","random.sample without replacement."),("freq","Counter counts occurrences."),
("merge_sort","Two-pointer merge with remainders."),("lru_get","LRU move-to-end on hit, None on miss.")]

def call(m,prompt,mt=100):
    d=json.dumps({"model":m,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0}).encode()
    for a in range(4):
        try:
            req=urllib.request.Request(URL,data=d,headers=HD)
            with urllib.request.urlopen(req,timeout=45) as r:r2=json.loads(r.read())
            return r2["choices"][0]["message"]["content"].strip(),r2.get("usage",{}).get("total_tokens",0)
        except Exception as e:
            if "429" in str(e):time.sleep((a+1)*6)
            elif a<3:time.sleep(2)
    return "ERROR",0

def sp(f):
    n,d=f
    return f"Encode into 16 floats. FUNC:{n} DESC:{d}. Scheme: dims0-3=category,4-7=input,8-11=output,12-15=complexity. Output ONLY JSON array of 16 floats. JSON:"

def rp(z,idxs):
    vs=json.dumps([round(x,4) for x in z])
    cs="\n".join([f"  {i+1}. {FUNCS[idx][0]} - {FUNCS[idx][1]}" for i,idx in enumerate(idxs)])
    return f"Identify from 16D vector.\nVECTOR:{vs}\nCANDIDATES:\n{cs}\nReply ONLY number 1-{N}. Ans:"

def game(sm,rm,null=False):
    idxs=np.random.choice(len(FUNCS),N,replace=False);tidx=np.random.randint(0,N)
    sr,st=call(sm,sp(FUNCS[idxs[tidx]]),100)
    if null:z=list(np.round(np.random.randn(LATENT_DIM)*0.5,4))
    else:
        try:m=re.search(r'\[[\d\s,.\-]+\]',sr);z=json.loads(m.group()) if m else[0]*LATENT_DIM
        except:z=[0]*LATENT_DIM
    rr,rt=call(rm,rp(z,idxs),10)
    try:nums=re.findall(r'\b([1-4])\b',rr)
    except:nums=[]
    return (int(nums[0])-1==tidx,st+rt) if nums else (False,st+rt)

if __name__=="__main__":
    print("NEURALESE MULTI-LLM GAME")
    for m in MODELS:print(f"  {m.split('/')[-1][:30]}")
    results=[];G=5
    for m in MODELS:
        sn=m.split("/")[-1][:14];print(f"[{sn}]",end=" ",flush=True)
        c=nc=0
        for i in range(G):ok,_=game(m,m);nok,_=game(m,m,null=True);c+=ok;nc+=nok
        acc=c/G;null=nc/G;results.append({"p":sn,"a":acc,"n":null,"o":acc-0.25})
        print(f"acc={acc:.0%} null={null:.0%} over={acc-0.25:+.0%}",flush=True)

    best=max(results,key=lambda r:r["a"]);bm=[m for m in MODELS if m.split("/")[-1][:14]==best["p"]][0]
    print(f"\n[Cross: {best['p'][:10]}->all]")
    for m in MODELS:
        if m==bm:continue;sn=m.split("/")[-1][:12]
        print(f"  ->{sn}",end=" ",flush=True);c=nc=0
        for i in range(G):ok,_=game(bm,m);nok,_=game(bm,m,null=True);c+=ok;nc+=nok
        acc=c/G;null=nc/G;results.append({"p":f"{best['p'][:10]}->{sn}","a":acc,"n":null,"o":acc-0.25})
        print(f"acc={acc:.0%} null={null:.0%}",flush=True)

    print(f"\n{'Pair':>25s} {'Acc':>6s} {'Null':>6s} {'Over':>6s} {'V':>6s}")
    for r in results:
        v="SIG" if r["o"]>0.05 else("STRONG" if r["o"]>0.25 else"NOISE")
        print(f"{r['p']:>25s} {r['a']:>5.0%} {r['n']:>5.0%} {r['o']:>5.0%} {v:>6s}")
    json.dump(results,open("output/llm_multi.json","w"),indent=2)
    print("\nSaved")
