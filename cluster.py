import argparse,json,math,os,re,random
STOP=set("a an and are as at by for from in is of on or the to with we our your".split())
def tokenize(text): return [w for w in re.findall(r"[a-z0-9]+",text.lower()) if w not in STOP]
def tfidf(documents):
    vocab=sorted(set(t for doc in documents for t in tokenize(doc)))
    if not vocab:return [{} for _ in documents]
    df={t:sum(t in set(tokenize(doc)) for doc in documents) for t in vocab};vectors=[]
    for doc in documents:
        terms=tokenize(doc); counts={t:terms.count(t) for t in set(terms)}; denom=max(1,len(terms))
        vec={t:(count/denom)*math.log((1+len(documents))/(1+df[t])) for t,count in counts.items()};norm=math.sqrt(sum(v*v for v in vec.values())) or 1
        vectors.append({t:v/norm for t,v in vec.items()})
    return vectors
def distance(a,b):return sum((a.get(k,0)-b.get(k,0))**2 for k in set(a)|set(b))
def cluster(documents,k,seed=1,iterations=60):
    if not documents or k<1 or k>len(documents):raise ValueError("k must be between 1 and document count")
    vectors=tfidf(documents);rng=random.Random(seed);centers=[dict(vectors[i]) for i in rng.sample(range(len(vectors)),k)];labels=[-1]*len(vectors)
    for _ in range(iterations):
        new=[min(range(k),key=lambda c:distance(v,centers[c])) for v in vectors]
        if new==labels:break
        labels=new
        for c in range(k):
            members=[vectors[i] for i,l in enumerate(labels) if l==c]
            if not members:continue
            keys=set(t for v in members for t in v);center={t:sum(v.get(t,0) for v in members)/len(members) for t in keys};norm=math.sqrt(sum(x*x for x in center.values())) or 1
            centers[c]={t:x/norm for t,x in center.items()}
    summaries=[]
    for c in range(k):
        terms={t:sum(v.get(t,0) for i,v in enumerate(vectors) if labels[i]==c) for t in set(t for i,v in enumerate(vectors) if labels[i]==c for t in v)}
        summaries.append({"cluster":c,"documents":[i for i,l in enumerate(labels) if l==c],"top_terms":[t for t,_ in sorted(terms.items(),key=lambda x:x[1],reverse=True)[:5]]})
    return labels,summaries
def main():
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--k',type=int,default=3);p.add_argument('--seed',type=int,default=1);p.add_argument('--out',default='clusters.json');a=p.parse_args()
    files=sorted(f for f in os.listdir(a.folder) if f.endswith('.txt'));docs=[open(os.path.join(a.folder,f),encoding='utf8').read() for f in files];labels,summaries=cluster(docs,a.k,a.seed)
    result={"documents":[{"file":f,"cluster":labels[i]} for i,f in enumerate(files)],"clusters":summaries};open(a.out,'w',encoding='utf8').write(json.dumps(result,indent=2));print(f"Clustered {len(files)} docs into {a.k} groups -> {a.out}")
if __name__=='__main__':main()
