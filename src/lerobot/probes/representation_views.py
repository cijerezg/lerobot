"""Native model capture sites and explicit selection of their cached views."""
from __future__ import annotations
import json
from pathlib import Path

VIEW_METADATA = {
    'block_output': {
        'label': 'Raw residual output',
        'encoder': 'Complete VLM block output, after attention, MLP and residual additions.',
        'action': 'Complete action-expert block output, averaged over future action positions.',
        'layers': 'Lℓ is the output of block ℓ (zero-based), before final network normalization.',
    },
    'attention_input': {
        'label': 'Native attention input',
        'encoder': 'Actual output of VLM block.attn_norm, including its learned per-feature scale.',
        'action': 'Actual input to expert cross-attention, after cross_norm and native time-conditioned shift/scale.',
        'layers': 'Lℓ identifies the consuming block ℓ (zero-based). VLM L17 normalizes the residual output of block 16; L0 normalizes the embedding input.',
    },
    'expert_key': {
        'label': 'Native expert keys',
        'encoder': 'Actual kv_k supplied to expert cross-attention: context projection, context RMS normalization and per-head key normalization already applied; heads flattened without changing values.',
        'action': None,
        'layers': 'Lℓ identifies expert block ℓ and the VLM block ℓ that produced its cached keys, before that VLM block’s MLP output.',
    },
    'expert_value': {
        'label': 'Native expert values',
        'encoder': 'Actual kv_v supplied to expert cross-attention, after context projection and context RMS normalization; heads flattened without changing values.',
        'action': None,
        'layers': 'Lℓ identifies expert block ℓ and the VLM block ℓ that produced its cached values, before that VLM block’s MLP output.',
    },
}
NATIVE_VIEWS = tuple(v for v in VIEW_METADATA if v != 'block_output')


def resolve_view_cache(cache_dir, view):
    """Never reinterpret a legacy/raw pooled cache as a normalized capture."""
    if view not in VIEW_METADATA:
        raise ValueError(f'Unknown representation view {view!r}; choose {tuple(VIEW_METADATA)}')
    root=Path(cache_dir)
    meta=json.loads((root/'meta.json').read_text())
    current=meta.get('representation_view','block_output')
    if current==view:return str(root)
    target=root/'native'/view
    if current=='block_output' and (target/'meta.json').exists():
        selected=json.loads((target/'meta.json').read_text())
        if selected.get('representation_view')!=view:
            raise ValueError(f'Native cache metadata does not match requested view {view}: {target}')
        if selected.get('rows')!=meta.get('rows'):
            raise ValueError(f'Native cache sampling rows do not match the parent cache: {target}; collect again.')
        if selected.get('capture_id')!=meta.get('capture_id'):
            raise ValueError(f'Native cache is from a different capture: {target}; collect again.')
        return str(target)
    raise ValueError(
        f'Cache {root} contains {current!r}, not native view {view!r}. '
        'Collect again with enable_subspace_spans=true (mode=collect or all), '
        'or explicitly select subspace_view=block_output / --view block_output for the existing raw cache. '
        'Native normalization cannot be recovered from pooled vectors.'
    )


class NativeRepresentationCapture:
    """Read native module outputs/arguments during the existing forward; never recompute a norm."""
    def __init__(self, transformer, expert, groups, device):
        import torch
        self.torch=torch
        self.groups=groups
        self.names=[g for g,ids in groups.items() if ids]
        self.indices=[torch.tensor(groups[g],device=device) for g in self.names]
        self.layers=len(transformer.blocks)
        self.values={v:[None]*self.layers for v in NATIVE_VIEWS}
        self.queries=[None]*self.layers
        self.handles=[]
        if len(expert.blocks)!=self.layers:
            raise ValueError('Native views require one expert block per VLM block.')
        try:
            for layer,(block,action_block) in enumerate(zip(transformer.blocks,expert.blocks)):
                self.handles.append(block.attn_norm.register_forward_hook(self._norm_hook(layer)))
                self.handles.append(action_block.cross_attn.register_forward_pre_hook(self._cross_hook(layer),with_kwargs=True))
        except Exception:
            self.close();raise

    def _pool(self,x):
        # Native K/V are [B,S,heads,head_dim]; flattening preserves every coordinate.
        row=x[0].detach().float().reshape(x.shape[1],-1)
        return self.torch.stack([row[ids].mean(0) for ids in self.indices])

    def _save(self,view,layer,x):
        if self.values[view][layer] is not None:
            raise RuntimeError(f'Duplicate native capture at {view}/L{layer}')
        self.values[view][layer]=self._pool(x)

    def _norm_hook(self,layer):
        def hook(module,args,out):self._save('attention_input',layer,out)
        return hook

    def _cross_hook(self,layer):
        def hook(module,args,kwargs):
            self._save('expert_key',layer,kwargs['kv_k'])
            self._save('expert_value',layer,kwargs['kv_v'])
            x=args[0] if args else kwargs['x']
            self.queries[layer]=x[0].detach().float().mean(0)
        return hook

    def result(self):
        result={}
        for view,values in self.values.items():
            if any(x is None for x in values):raise RuntimeError(f'Incomplete native capture: {view}')
            pooled=self.torch.stack(values).cpu()
            encoder=dict.fromkeys(self.groups)
            for i,g in enumerate(self.names):encoder[g]=pooled[:,i]
            result[view]={'encoder':encoder,'action_expert':{},'n_tokens':{g:len(v) for g,v in self.groups.items()}}
        if any(x is None for x in self.queries):raise RuntimeError('Incomplete expert attention-input capture')
        result['attention_input']['action_expert']['action']=self.torch.stack(self.queries).cpu()
        return result

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]
