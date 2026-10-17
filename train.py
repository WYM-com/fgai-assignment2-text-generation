"""Reproducible word-level autoregressive comparison; run python train.py."""
import argparse, collections, csv, hashlib, json, math, platform, random, re, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace('’', "'"))

class LanguageModel(nn.Module):
    def __init__(self, kind, vocab_size, embedding=64, hidden=96):
        super().__init__()
        self.kind=kind
        self.embedding=nn.Embedding(vocab_size, embedding)
        cls={'RNN':nn.RNN,'BiRNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[kind]
        self.recurrent=cls(embedding,hidden,batch_first=True,bidirectional=kind=='BiRNN')
        self.output=nn.Linear(hidden*(2 if kind=='BiRNN' else 1),vocab_size)
    def forward(self,x):
        _,h=self.recurrent(self.embedding(x))
        if self.kind=='LSTM': h=h[0]
        # Both directions see only the supplied PAST context, never the target.
        h=torch.cat([h[-2],h[-1]],dim=1) if self.kind=='BiRNN' else h[-1]
        return self.output(h)

def sequences(tokens, vocab, context):
    ids=torch.tensor([vocab.get(t,0) for t in tokens],dtype=torch.long)
    windows=ids.unfold(0,context+1,1).contiguous()
    return TensorDataset(windows[:,:-1],windows[:,-1])

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); total=correct=n=0
    for x,y in loader:
        logits=model(x)
        total+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
        correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
    return {'loss':total/n,'perplexity':math.exp(total/n),'accuracy':correct/n}

@torch.no_grad()
def generate(model,vocab,seed,temperature,context,length,rng_seed):
    model.eval(); words=list(vocab); ids=[vocab.get(t,0) for t in tokenize(seed)]
    generator=torch.Generator().manual_seed(rng_seed); output=[]
    for _ in range(length):
        logits=model(torch.tensor([ids[-context:]],dtype=torch.long))[0]/temperature
        logits[0]=-float('inf') # Do not print the unknown-token placeholder.
        nxt=torch.multinomial(torch.softmax(logits,0),1,generator=generator).item()
        ids.append(nxt); output.append(words[nxt])
    bigrams=list(zip(output,output[1:])); trigrams=list(zip(output,output[1:],output[2:]))
    return ' '.join(output),len(set(output))/len(output),len(set(bigrams))/len(bigrams),1-len(set(trigrams))/len(trigrams)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--seed',type=int,default=42); args=parser.parse_args()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    results=ROOT/'results'; results.mkdir(exist_ok=True); (ROOT/'checkpoints').mkdir(exist_ok=True)
    raw=(ROOT/'data/alice.txt').read_text(encoding='utf-8')
    body=re.split(r'\*\*\* START OF .*?\*\*\*',raw,maxsplit=1)[1]
    body=re.split(r'\*\*\* END OF .*?\*\*\*',body,maxsplit=1)[0]
    # Remove the table of contents; retain narrative beginning at chapter I.
    starts=list(re.finditer(r'CHAPTER I\.\s*Down the Rabbit-Hole',body))
    assert starts, 'Expected chapter heading missing'
    body=body[starts[-1].start():]
    tokens=tokenize(body); a=int(.8*len(tokens)); b=int(.9*len(tokens))
    splits=[tokens[:a],tokens[a:b],tokens[b:]]
    counts=collections.Counter(splits[0]); words=['<unk>']+sorted(w for w,c in counts.items() if c>=2)
    vocab={w:i for i,w in enumerate(words)}; context=20
    datasets=[sequences(s,vocab,context) for s in splits]
    metadata={'source':'https://www.gutenberg.org/ebooks/11','download':'https://www.gutenberg.org/ebooks/11.txt.utf-8',
      'sha256':hashlib.sha256((ROOT/'data/alice.txt').read_bytes()).hexdigest(),'tokens':len(tokens),'unique_raw_words':len(set(tokens)),
      'vocabulary':len(vocab),'split_tokens':[len(s) for s in splits],'split_examples':[len(s) for s in datasets],
      'oov_rates':[sum(w not in vocab for w in s)/len(s) for s in splits],
      'context':context,'embedding':64,'hidden':96,'batch_size':128,'epochs':args.epochs,'learning_rate':.001,
      'seed':args.seed,'device':'CPU','threads':4,'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__}
    (results/'metadata.json').write_text(json.dumps(metadata,indent=2)); (results/'vocabulary.json').write_text(json.dumps(vocab,indent=2))
    eval_loaders=[DataLoader(d,batch_size=256) for d in datasets[1:]]
    rows=[]; history=[]; samples=[]
    seeds=['alice was beginning to get very tired','the queen said to alice','the rabbit took a watch out of']
    for kind in ['RNN','BiRNN','LSTM','GRU']:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        loader=DataLoader(datasets[0],batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        model=LanguageModel(kind,len(vocab)); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        best=float('inf'); best_epoch=0; started=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            model.train(); total=n=0
            for x,y in loader:
                optimizer.zero_grad(); loss=nn.functional.cross_entropy(model(x),y); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                total+=loss.item()*len(y); n+=len(y)
            val=evaluate(model,eval_loaders[0]); history.append({'model':kind,'epoch':epoch,'train_loss':total/n,'val_loss':val['loss']})
            if val['loss']<best:
                best=val['loss']; best_epoch=epoch
                torch.save(model.state_dict(),ROOT/'checkpoints'/f'{kind}.pt')
            print(f'{kind} epoch {epoch:02d}: train={total/n:.4f} val={val["loss"]:.4f}',flush=True)
        elapsed=time.perf_counter()-started
        model.load_state_dict(torch.load(ROOT/'checkpoints'/f'{kind}.pt',weights_only=True))
        row={'model':kind,**evaluate(model,eval_loaders[1]),'training_seconds':elapsed,'best_epoch':best_epoch,'best_val_loss':best,'parameters':sum(p.numel() for p in model.parameters())}
        rows.append(row)
        for seed in seeds:
            for temp in [.5,1.,1.5]:
                for repeat in range(3):
                    continuation,d1,d2,r3=generate(model,vocab,seed,temp,context,50,1000+repeat)
                    samples.append({'model':kind,'seed':seed,'temperature':temp,'repeat':repeat,'continuation':continuation,'distinct_1':d1,'distinct_2':d2,'repeated_trigram_fraction':r3})
        (results/'metrics.json').write_text(json.dumps(rows,indent=2))
    for filename,data in [('metrics.csv',rows),('history.csv',history),('generated_samples.csv',samples)]:
        with (results/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (results/'generated_samples.json').write_text(json.dumps(samples,indent=2))
    summary=[]
    for kind in ['RNN','BiRNN','LSTM','GRU']:
        for temp in [.5,1.,1.5]:
            subset=[s for s in samples if s['model']==kind and s['temperature']==temp]
            summary.append({'model':kind,'temperature':temp,**{k:float(np.mean([s[k] for s in subset])) for k in ['distinct_1','distinct_2','repeated_trigram_fraction']}})
    (results/'temperature_summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for kind in ['RNN','BiRNN','LSTM','GRU']:
        h=[r for r in history if r['model']==kind]
        for ax,key in zip(axes,['train_loss','val_loss']): ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=kind)
    for ax,title in zip(axes,['Training loss','Validation loss']): ax.set(title=title,xlabel='Epoch',ylabel='Cross entropy'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(results/'learning_curves.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,['perplexity','accuracy','training_seconds'],['Test perplexity (lower better)','Test accuracy (fraction)','Training time (seconds)']):
        ax.bar([r['model'] for r in rows],[r[key] for r in rows],color=['#4979a9','#63a6a0','#e4b158','#b87878']); ax.set_title(title)
    fig.tight_layout(); fig.savefig(results/'comparison.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for kind in ['RNN','BiRNN','LSTM','GRU']:
        s=[r for r in summary if r['model']==kind]
        for ax,key in zip(axes,['distinct_1','repeated_trigram_fraction']): ax.plot([r['temperature'] for r in s],[r[key] for r in s],marker='o',label=kind)
    for ax,title in zip(axes,['Distinct word fraction','Repeated trigram fraction']): ax.set(title=title,xlabel='Temperature',ylabel='Mean over 9 samples'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(results/'temperature.png',dpi=170)
    print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__': main()
