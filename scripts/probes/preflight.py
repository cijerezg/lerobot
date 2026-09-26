"""Read-only configured data/cache gate; does not instantiate a model or train."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch,yaml
from lerobot.datasets.diverse_actor_selection import open_federated_corpus,select_actor_anchors,holdout_episode_ids
from lerobot.rl.data_sources.diverse_actor_buffer import DiverseSampleSpec
from lerobot.rl.data_sources.diverse_actor_cache import find_cache
from lerobot.rl.buffer import ReplayBuffer
p=argparse.ArgumentParser();p.add_argument('--config',default='config_rl_validate.yaml');args=p.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text());root=Path(cfg['diverse']['root'])
selection=select_actor_anchors(open_federated_corpus(root));held=holdout_episode_ids(root)
assert len(held)==18 and not set(selection.episode_ids)&held
assert len(selection.rows)==87306 and len(selection.episode_ids)==891
stats=torch.load(cfg['policy']['embodiment_stats_path'],map_location='cpu',weights_only=False)
assert set(stats['probe_holdout'])==held
for i,count in ((0,5586),(1,7401),(2,11189),(3,22618),(4,17533),(5,954),(7,21282),(8,743)):
    assert int(stats['counts'][i,0])==count,(i,stats['counts'][i])
cache=find_cache(root,cfg['diverse']['cache_dir'],DiverseSampleSpec(),anchors=len(selection.rows),episodes=len(selection.episode_ids))
assert cache is not None,'Configured diverse cache is missing'
meta=json.loads((cache/'metadata.json').read_text());assert meta['built_rows']==87306 and not meta['partial']
sources=cfg['dataset']['sources']
# The ReBot stats pool is matched by data (frame + episode counts), not by path: a re-keyed
# root (hardlink twin under a new name) is the same data. Sources the pool never saw are
# listed, not fatal: with a pretrained_path that carries processors, normalization comes
# from the checkpoint and the stats file is not read (MolmoAct2Trainer, "stats source").
def _counts(r):
    info=json.loads((Path(r)/'meta/info.json').read_text());return (info['total_frames'],info['total_episodes'])
stats_roots=stats['roots']['rebot_b601_joint7_commanded'];source_counts={_counts(s['root']):s['root'] for s in sources}
unmatched=[r for r in stats_roots if _counts(r) not in source_counts];assert not unmatched,f'stats roots without a same-size source: {unmatched}'
pooled={_counts(r) for r in stats_roots};outside_pool=[s['root'] for s in sources if _counts(s['root']) not in pooled]
for s in sources:
    r=Path(s['root']);info=json.loads((r/'meta/info.json').read_text())
    proxy=SimpleNamespace(root=r,meta=SimpleNamespace(total_frames=info['total_frames'],total_episodes=info['total_episodes']))
    keys=[k for k in info['features'] if k.startswith('observation.')]
    fp=ReplayBuffer._dataset_fingerprint(proxy,state_keys=keys,image_storage_dtype=cfg['policy'].get('image_storage_dtype','uint8'),image_storage_size=cfg['policy'].get('image_storage_size'),image_stride=cfg['policy'].get('image_stride',3))
    path=Path(cfg['buffer_cache_dir'])/fp/'metadata.json';assert path.is_file(),path
    metadata=json.loads(path.read_text());assert metadata['num_transitions']==info['total_frames']
val=Path(cfg['val_dataset_path']);vi=json.loads((val/'meta/info.json').read_text());assert vi['total_episodes']>=cfg['probe_parameters']['max_episodes']
assert Path(cfg['probe_parameters']['subtask_scene_sweep_frames']).is_file()
print(f'PASS: {len(sources)} ReBot sources ({len(outside_pool)} outside the stats pool: {outside_pool}), {vi["total_episodes"]} validation episodes; 891 diverse training episodes / 87306 anchors; configured stats and caches agree.')
