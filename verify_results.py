"""Re-evaluate saved checkpoints and audit experiment artifacts."""
import csv,json,math,re
import torch
from torch.utils.data import DataLoader
from train import ROOT,LanguageModel,tokenize,sequences,evaluate

torch.set_num_threads(4)
r=ROOT/'results'; meta=json.loads((r/'metadata.json').read_text()); vocab=json.loads((r/'vocabulary.json').read_text())
raw=(ROOT/'data/alice.txt').read_text()
body=re.split(r'\*\*\* START OF .*?\*\*\*',raw,maxsplit=1)[1]
body=re.split(r'\*\*\* END OF .*?\*\*\*',body,maxsplit=1)[0]
start=list(re.finditer(r'CHAPTER I\.\s*Down the Rabbit-Hole',body))[-1].start()
tokens=tokenize(body[start:]); a=int(.8*len(tokens)); b=int(.9*len(tokens))
assert len(tokens)==meta['tokens']
assert set(vocab)-{'<unk>'} <= set(tokens[:a]), 'Vocabulary leaked held-out words'
test=DataLoader(sequences(tokens[b:],vocab,20),batch_size=256)
for row in json.loads((r/'metrics.json').read_text()):
    model=LanguageModel(row['model'],len(vocab))
    model.load_state_dict(torch.load(ROOT/'checkpoints'/f'{row["model"]}.pt',weights_only=True))
    actual=evaluate(model,test)
    for key in actual: assert math.isclose(actual[key],row[key],rel_tol=1e-6),(row['model'],key)
    assert math.isclose(math.exp(row['loss']),row['perplexity'])
samples=json.loads((r/'generated_samples.json').read_text())
assert len(samples)==108
assert len({(s['model'],s['seed'],s['temperature'],s['repeat']) for s in samples})==108
assert all(len(s['continuation'].split())==50 for s in samples)
with (r/'history.csv').open() as f: assert len(list(csv.DictReader(f)))==60
print('PASS: all four checkpoint metrics reproduced; training-only vocabulary, 60 epochs and 108 fifty-word continuations verified.')
