"""Per-head scores on an identical frozen candidate set; frequency is a hypothesis."""
from __future__ import annotations
import numpy as np
import torch
from tempokv.spectral.sensitivity import Block, attention_mass
from .capture import stable_seed
from .metrics import standardize


def cosine(a,b,eps=1e-12):
    a,b=a.float().flatten(),b.float().flatten();den=a.norm()*b.norm()
    return float((a@b)/den) if den>eps else float("nan")


def candidate_scores(window,candidates,oracle_rows):
    count=len(candidates); nearest={};similarity=np.full(count,np.nan);residual=np.full(count,np.nan)
    for i,(a,indices) in enumerate(candidates):
        possible=[]
        for j,(b,other) in enumerate(candidates):
            if i==j or len(indices)!=len(other):continue
            ck,cv=cosine(a.k_pre,b.k_pre),cosine(a.v,b.v)
            if np.isfinite(ck+cv):possible.append(((ck+cv)/2,j,ck,cv))
        if possible:
            value,j,ck,cv=max(possible,key=lambda x:(x[0],-x[1]));nearest[i]=(j,ck,cv)
            similarity[i]=value;residual[i]=1-value
    coarse_dim=min(16,window.q_post.shape[-1]);scale=(coarse_dim/window.q_post.shape[-1])**.5
    coarse=attention_mass(window.q_post[...,:coarse_dim].float()*scale,window.k_post[...,:coarse_dim].float(),
                          [Block(a.frame,a.block_id,indices) for a,indices in candidates])
    frequency=np.asarray([(a.freq_k_high+a.freq_v_high)/2 for a,_ in candidates])
    random=np.random.default_rng(stable_seed(window.context.identity,"random_score")).random(count)
    values={"frequency":frequency,"similarity":similarity,"residual":residual,"attention":np.asarray([r["attention_mass"] for r in oracle_rows]),
            "coarse_qk":np.asarray(coarse),"frequency_qk":.5*standardize(frequency)+.5*standardize(coarse),"random":random}
    return values,nearest
