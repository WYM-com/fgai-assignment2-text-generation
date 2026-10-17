"""Generate from the saved best-validation checkpoint without retraining."""
import argparse,json
import torch
from train import ROOT,LanguageModel,generate
p=argparse.ArgumentParser()
p.add_argument('--model',choices=['RNN','BiRNN','LSTM','GRU'],default='GRU')
p.add_argument('--seed',default='alice was beginning to get very tired')
p.add_argument('--temperature',type=float,default=1.0)
p.add_argument('--words',type=int,default=50)
p.add_argument('--random-seed',type=int,default=1000)
a=p.parse_args()
if a.temperature<=0 or a.words<3: p.error('temperature must be positive and words must be at least 3')
if not a.seed.strip(): p.error('seed cannot be empty')
torch.set_num_threads(4)
vocab=json.loads((ROOT/'results/vocabulary.json').read_text())
model=LanguageModel(a.model,len(vocab))
model.load_state_dict(torch.load(ROOT/'checkpoints'/f'{a.model}.pt',map_location='cpu',weights_only=True))
print(a.seed+' '+generate(model,vocab,a.seed,a.temperature,20,a.words,a.random_seed)[0])
