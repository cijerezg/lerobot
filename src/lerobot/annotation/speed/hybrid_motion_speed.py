"""Smooth measured motion, with a bounded adjustment from duration-based pace.

The speed scale remains anchored to v5 training states. Duration can move the
continuous score by at most 0.35 of one bucket; task quality is never an input.
frame_runs turns the per-frame score into the frame-level label the trainer reads
(2026-10-26): the rounded score, with no run shorter than MIN_RUN_SECONDS.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

METHOD='hybrid_motion_duration_v1'
LOCAL_SECONDS=1.0
DURATION_WEIGHT=0.2
MAX_DURATION_ADJUSTMENT=0.35
FINAL_SECONDS=0.4
# One action chunk (30 steps at 30 Hz): a shorter run is absorbed by a neighbour.
MIN_RUN_SECONDS=1.0


def quantize(scores):
    return np.clip(np.floor(np.asarray(scores)+.5),1,5).astype(np.uint8)


def motion_scores(values, sorted_training_values):
    values=np.asarray(values,dtype=float)
    ref=np.asarray(sorted_training_values,dtype=float)
    if len(ref)<100 or not np.isfinite(ref).all() or ref[-1]<=ref[0]:
        raise ValueError('Insufficient motion reference')
    percentile=np.searchsorted(ref,values,side='right')/len(ref)
    return np.clip(.5+5*percentile,1.,5.)


def blend_scores(motion_score, duration_speed, use_duration=True):
    score=np.asarray(motion_score,dtype=float)
    correction=np.clip(DURATION_WEIGHT*(duration_speed-score),-MAX_DURATION_ADJUSTMENT,MAX_DURATION_ADJUSTMENT) if use_duration else 0.
    return np.clip(score+correction,1.,5.)


def hybrid_trace(q, rate, atoms, sorted_training_values):
    """Atoms specify start/end, duration speed and whether that prior is usable.

    Arrays index t -> t+1. Both smoothing stages stop at excluded gaps. Invalid
    runs remain unscorable; the caller assigns default 3 only inside supervision.
    """
    q=np.asarray(q,dtype=float)
    if q.ndim!=2 or not np.isfinite(rate) or rate<=0:
        raise ValueError('Expected joint states and finite positive native rate')
    raw=np.linalg.norm(np.diff(q,axis=0),axis=1)*rate
    n=len(raw);atoms=sorted(atoms,key=lambda r:r['start_timestep'])
    local=np.full(n,np.nan);score=np.full(n,np.nan);motion=np.full(n,np.nan);prior=np.full(n,np.nan)
    mask=np.zeros(n,dtype=bool);runs=[];previous_end=0
    for atom in atoms:
        a,b=atom['start_timestep'],atom['end_timestep_exclusive']
        if not 0<=a<b<=len(q) or a<previous_end:raise ValueError('Invalid or overlapping atom spans')
        previous_end=b;mask[a:b-1]=True
        if runs and runs[-1][-1]['end_timestep_exclusive']==a:runs[-1].append(atom)
        else:runs.append([atom])
    for run in runs:
        start,end=run[0]['start_timestep'],run[-1]['end_timestep_exclusive']-1
        if end<=start or not np.isfinite(raw[start:end]).all():continue
        local[start:end]=np.maximum(0,uniform_filter1d(raw[start:end],size=max(1,round(LOCAL_SECONDS*rate)),mode='nearest'))
        motion[start:end]=motion_scores(local[start:end],sorted_training_values)
        for atom in run:
            a,b=atom['start_timestep'],min(atom['end_timestep_exclusive'],end)
            score[a:b]=blend_scores(motion[a:b],atom['duration_speed'],atom['use_duration'])
            prior[a:b]=atom['duration_speed']
        score[start:end]=uniform_filter1d(score[start:end],size=max(1,round(FINAL_SECONDS*rate)),mode='nearest')
        motion[start:end]=uniform_filter1d(motion[start:end],size=max(1,round(FINAL_SECONDS*rate)),mode='nearest')
    return {'joint_speed_rad_s':local,'motion_score':motion,'duration_score':prior,'speed_score':score,
            'valid':np.isfinite(score),'supervision_mask':mask}


def frame_runs(scores, rate):
    """Frame-level label of one contiguous stretch as runs [(start, end_exclusive, speed, mean_score)].

    scores holds one continuous score per frame (the trace's t -> t+1 score sits on frame
    t); a frame without a finite score counts as 3, the default for unscorable motion. Each
    frame takes quantize(score); then, shortest first, every run under MIN_RUN_SECONDS takes
    the label of the neighbouring run closer to its mean score, so no run is shorter than
    one action chunk unless the whole stretch is.
    """
    x=np.where(np.isfinite(scores),scores,3.)
    label=quantize(x).astype(int);n=max(1,round(MIN_RUN_SECONDS*rate))
    while True:
        cuts=np.r_[0,np.flatnonzero(np.diff(label))+1,len(label)];lengths=np.diff(cuts)
        if len(lengths)==1 or lengths.min()>=n:break
        i=int(np.argmin(lengths));a,b=cuts[i],cuts[i+1]
        neighbours=([label[a-1]] if i>0 else [])+([label[b]] if i<len(lengths)-1 else [])
        label[a:b]=min(neighbours,key=lambda v:abs(v-x[a:b].mean()))
    return [(int(a),int(b),int(label[a]),float(x[a:b].mean())) for a,b in zip(cuts[:-1],cuts[1:])]
