"""Prepare two independent RGB+Z/CVAE ACT runs for the idle-GPU queue."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from prepare_act_coarse_1020 import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True, help='Local project containing the source datasets')
    p.add_argument('--runtime-project', type=Path, required=True, help='Project path on the training server')
    args = p.parse_args()
    root = args.project.resolve()
    remote = args.runtime_project
    if not remote.is_absolute():
        raise ValueError('runtime-project must be absolute')
    config_dir = root / 'configs/act_fine_1027'
    config_dir.mkdir(parents=True, exist_ok=True)
    template = json.loads((root / 'configs/act_1023_z/train_config_cvae.json').read_text())
    plan = dict(host='10.0.40.163', status='PREPARED', poll_seconds=120, idle_samples=5,
                batch_size=32, steps=50000, chunk_size=4, use_cvae=True, runs=[])
    for source_rel, prepared_rel, name in [
        ('datasets/vga_fine/act_dataset_1023', 'datasets/vga_fine/vga_1023_z_1027', 'vga_1023_z_cvae_50k'),
        ('datasets/usb_fine/usb_dataset_1027', 'datasets/usb_fine/usb_1027_z', 'usb_1027_z_cvae_50k'),
    ]:
        source, prepared = root / source_rel, root / prepared_rel
        if not prepared.exists():
            prepare(source, prepared, 'rgb_z_delta_tcp')
        provenance = json.loads((prepared / 'meta/training_conversion.json').read_text())
        assert provenance['source_info_sha256'] == hashlib.sha256((source / 'meta/info.json').read_bytes()).hexdigest(), 'Source changed; use a new converted dataset path'
        info = json.loads((prepared / 'meta/info.json').read_text())
        assert info['features']['observation.state']['names'] == ['ee_z']
        assert info['features']['action']['names'] == ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
        cfg = deepcopy(template)
        cfg['dataset'].update(repo_id=prepared.name, root=str(remote / prepared_rel), episodes=None)
        h, w, c = info['features']['observation.images.wrist']['shape']
        cfg['policy']['input_features']['observation.images.wrist']['shape'] = [c, h, w]
        cfg.update(output_dir=str(remote / 'outputs/act_fine' / name), job_name=name,
                   batch_size=32, steps=50000, save_freq=50000, eval_freq=0)
        cfg['scheduler']['num_decay_steps'] = 50000
        path = config_dir / f'{name}.json'
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        plan['runs'].append(dict(name=name, source=source_rel, dataset=prepared_rel,
                                episodes=info['total_episodes'], frames=info['total_frames'],
                                config=str(path.relative_to(root))))
    (config_dir / 'training_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2), flush=True)


if __name__ == '__main__':
    main()
