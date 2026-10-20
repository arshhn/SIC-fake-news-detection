"""WELFake binary news classification with a NumPy CNN and TF-IDF baseline."""
from pathlib import Path
import json, re, time, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.metrics import ConfusionMatrixDisplay

OUT=Path(__file__).resolve().parent
SEED=42
np.random.seed(SEED)

def clean(s):
    s='' if s is None or (isinstance(s,float) and np.isnan(s)) else str(s)
    return re.sub(r'\s+',' ',s).strip()

def make_vocab(texts,max_words=12000):
    from collections import Counter
    c=Counter()
    for t in texts: c.update(re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+",t.lower()))
    vocab={'<PAD>':0,'<UNK>':1}
    for w,_ in c.most_common(max_words-2): vocab[w]=len(vocab)
    return vocab

def encode(texts,vocab,length=300):
    out=np.zeros((len(texts),length),dtype=np.int32)
    for i,t in enumerate(texts):
        ids=[vocab.get(w,1) for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+",t.lower())[:length]]
        out[i,:len(ids)]=ids
    return out

class Adam:
    def __init__(self,p,lr=.003): self.p=p; self.lr=lr; self.t=0; self.m={k:np.zeros_like(v) for k,v in p.items()}; self.v={k:np.zeros_like(v) for k,v in p.items()}
    def step(self,g):
        self.t+=1
        for k,p in self.p.items():
            x=np.clip(g[k],-2,2); self.m[k]=.9*self.m[k]+.1*x; self.v[k]=.999*self.v[k]+.001*x*x
            p-=self.lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)

class TextCNN:
    """Embedding -> parallel width-3/4/5 convolutions -> global max pool -> sigmoid."""
    def __init__(self,vocab_size,dim=32,filters=24,seed=42):
        r=np.random.default_rng(seed); self.p={'E':r.normal(0,.08,(vocab_size,dim)).astype(np.float32)}; self.widths=[3,4,5]; self.filters=filters
        for w in self.widths:
            self.p[f'W{w}']=r.normal(0,np.sqrt(2/(w*dim)),(w,dim,filters)).astype(np.float32); self.p[f'b{w}']=np.zeros(filters,dtype=np.float32)
        self.p['O']=r.normal(0,.08,(filters*len(self.widths),1)).astype(np.float32); self.p['ob']=np.zeros(1,dtype=np.float32); self.opt=Adam(self.p)
    def forward(self,x,training=False):
        emb=self.p['E'][x]; B,L,D=emb.shape; pools=[]; caches=[]
        for w in self.widths:
            windows=np.stack([emb[:,j:j+w,:] for j in range(L-w+1)],axis=1)
            z=np.einsum('blwd,wdf->blf',windows,self.p[f'W{w}'])+self.p[f'b{w}']; act=np.maximum(z,0)
            idx=np.argmax(act,axis=1); pool=act[np.arange(B)[:,None],idx,np.arange(self.filters)[None,:]]
            pools.append(pool); caches.append((windows,z,idx))
        feat=np.concatenate(pools,axis=1); logits=(feat@self.p['O']+self.p['ob']).reshape(-1); prob=1/(1+np.exp(-np.clip(logits,-30,30)))
        return prob,(x,feat,caches)
    def batch_loss(self,x,y,update=False):
        prob,c=self.forward(x); eps=1e-7; loss=-np.mean(y*np.log(prob+eps)+(1-y)*np.log(1-prob+eps))
        if not update:return float(loss)
        x,feat,caches=c; B=len(y); dl=((prob-y)/B).reshape(B,1); g={k:np.zeros_like(v) for k,v in self.p.items()}
        g['O']=feat.T@dl; g['ob']=dl.sum(0); df=(dl@self.p['O'].T).reshape(B,len(self.widths),self.filters); demb=np.zeros((B,x.shape[1],self.p['E'].shape[1]),dtype=np.float32)
        for wi,w in enumerate(self.widths):
            windows,z,idx=caches[wi]; dz=np.zeros_like(z); dz[np.arange(B)[:,None],idx,np.arange(self.filters)[None,:]]=df[:,wi,:]*(z[np.arange(B)[:,None],idx,np.arange(self.filters)[None,:]]>0)
            g[f'W{w}']=np.einsum('blwd,blf->wdf',windows,dz); g[f'b{w}']=dz.sum((0,1)); dw=np.einsum('blf,wdf->blwd',dz,self.p[f'W{w}'])
            for j in range(w): demb[:,j:j+dw.shape[1],:]+=dw[:,:,j,:]
        np.add.at(g['E'],x.reshape(-1),demb.reshape(-1,demb.shape[-1]))
        self.opt.step(g); return float(loss)
    def predict_proba(self,x,batch=64):
        return np.concatenate([self.forward(x[i:i+batch])[0] for i in range(0,len(x),batch)])

def score(y,p):
    pred=(p>=.5).astype(int); pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(pr),'recall':float(re),'f1':float(f1),'roc_auc':float(roc_auc_score(y,p)),'confusion_matrix':confusion_matrix(y,pred).tolist()}

def main():
    path=OUT/'WELFake.parquet'
    if not path.exists(): raise FileNotFoundError('Place WELFake.parquet beside this script; download instructions are in README.md')
    df=pd.read_parquet(path); df['title']=df.title.map(clean); df['text']=df.text.map(clean); df['label']=pd.to_numeric(df.label,errors='coerce')
    df=df.dropna(subset=['label']); df['label']=df.label.astype(int); df=df[(df.label.isin([0,1])) & ((df.title+' '+df.text).str.len()>20)].drop_duplicates(subset=['title','text']).reset_index(drop=True)
    texts=df.title.tolist(); labels=df.label.to_numpy(dtype=np.int32)
    train_idx,test_idx=train_test_split(np.arange(len(labels)),test_size=.15,stratify=labels,random_state=SEED)
    train_idx,val_idx=train_test_split(train_idx,test_size=.1765,stratify=labels[train_idx],random_state=SEED)
    ytr,yva,yte=labels[train_idx],labels[val_idx],labels[test_idx]
    print(f'Dataset: {len(labels):,} articles | fake={int((labels==0).sum()):,} real={int((labels==1).sum()):,}')
    print(f'Split: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}; label 0=fake, 1=real')
    # Vocabulary learned exclusively from training documents.
    # Vocabulary learned exclusively from train split. CNN sees capped leading 300 tokens.
    vocab=make_vocab([texts[i] for i in train_idx]); X=encode(texts,vocab,48)
    np.savez_compressed(OUT/'vocabulary.npz',words=np.array(list(vocab.keys()),dtype=str))
    model=TextCNN(len(vocab)); epochs=8; batch=128; history=[]; best_loss=float('inf'); best=None; rng=np.random.default_rng(SEED)
    for ep in range(1,epochs+1):
        start=time.time(); order=rng.permutation(train_idx); losses=[]
        for j in range(0,len(order),batch):
            ix=order[j:j+batch]; losses.append(model.batch_loss(X[ix],labels[ix],True))
        # Validation loss evaluated in batches
        vl=[]
        for j in range(0,len(val_idx),batch):
            ix=val_idx[j:j+batch]; vl.append(model.batch_loss(X[ix],labels[ix]))
        tr=float(np.mean(losses)); va=float(np.mean(vl)); history.append({'epoch':ep,'train_loss':tr,'val_loss':va,'seconds':round(time.time()-start,1)})
        print(f"Epoch {ep:02}/{epochs} | train loss={tr:.4f} | val loss={va:.4f} | {history[-1]['seconds']:.1f}s",flush=True)
        if va<best_loss: best_loss=va; best={k:v.copy() for k,v in model.p.items()}
    model.p=best
    deep_probs=model.predict_proba(X[test_idx]); cnn_score=score(yte,deep_probs)
    # Final scores on untouched test set.
    cnn_score['classification_report']=classification_report(yte,(deep_probs>=.5).astype(int),target_names=['fake','real'],output_dict=True,zero_division=0)
    np.savez_compressed(OUT/'cnn_checkpoint.npz',**model.p)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot([h['epoch'] for h in history],[h['train_loss'] for h in history],marker='o',label='Training'); ax.plot([h['epoch'] for h in history],[h['val_loss'] for h in history],marker='o',label='Validation'); ax.set(xlabel='Epoch',ylabel='Binary cross-entropy',title='TextCNN training'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'training_curve.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(confusion_matrix(yte,(deep_probs>=.5).astype(int)),display_labels=['Fake','Real']).plot(ax=ax,colorbar=False); ax.set_title('TextCNN test set'); fig.tight_layout(); fig.savefig(OUT/'confusion_matrices.png',dpi=160); plt.close(fig)
    result={'dataset':'WELFake','source':'https://huggingface.co/datasets/davanstrien/WELFake','input':'headline only, capped at 48 tokens','seed':SEED,'rows_after_cleaning':len(labels),'label_mapping':{'0':'fake','1':'real'},'split':{'train':len(train_idx),'validation':len(val_idx),'test':len(test_idx)},'cnn':{'architecture':'parallel convolution widths 3, 4, 5; embedding 32; 24 filters each; global max pooling; sigmoid','epochs':history,'best_val_loss':best_loss,'test':cnn_score}}
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # A few test examples with true label and model probabilities; deterministic order.
    rows=[]
    for ix,p in zip(test_idx[:8],deep_probs[:8]): rows.append({'actual':'real' if labels[ix]==1 else 'fake','predicted':'real' if p>=.5 else 'fake','probability_real':round(float(p),4),'title':df.iloc[ix].title[:240]})
    (OUT/'example_predictions.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print('TextCNN:',json.dumps({k:round(v,4) for k,v in cnn_score.items() if isinstance(v,float)}))
    print(f'Saved results, checkpoints, plots, and examples to {OUT}')

if __name__=='__main__': main()
