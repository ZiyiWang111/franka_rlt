#!/usr/bin/env python3
"""Prepare action6 addon data and aggregate it with the existing camera-only set."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.aggregate import aggregate_datasets
from scripts.prepare_act_camera_only_action6_tmp import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--addon', type=Path, required=True)
    parser.add_argument('--prepared-addon', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, addon, prepared, output = [p.resolve() for p in (
        args.base, args.addon, args.prepared_addon, args.output)]
    if output.exists() or prepared.exists():
        raise FileExistsError('Prepared addon and merged output must both be new directories')
    original_hashes = {str(p): hashlib.sha256((p / 'meta/info.json').read_bytes()).hexdigest()
                       for p in (base, addon)}
    base_info = json.loads((base / 'meta/info.json').read_text())
    assert 'observation.state' not in base_info['features']
    info = prepare(addon, prepared)
    # Preserve the base schema; extra force/torque columns never enter training.
    extra = set(info['features']) - set(base_info['features'])
    for file in (prepared / 'data').rglob('*.parquet'):
        table = pq.read_table(file)
        table = table.drop([key for key in extra if key in table.column_names])
        pq.write_table(table.replace_schema_metadata(None), file)
    for file in (prepared / 'meta/episodes').rglob('*.parquet'):
        table = pq.read_table(file)
        table = table.drop([key for key in table.column_names
                            if any(key.startswith(f'stats/{feature}/') for feature in extra)])
        pq.write_table(table.replace_schema_metadata(None), file)
    stats_path = prepared / 'meta/stats.json'
    stats = json.loads(stats_path.read_text())
    for key in extra:
        info['features'].pop(key)
        stats.pop(key, None)
    (prepared / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')
    stats_path.write_text(json.dumps(stats, indent=2) + '\n')
    if info['features'] != base_info['features']:
        raise ValueError('Prepared addon schema differs from the original camera-only schema')
    with TemporaryDirectory(prefix='.merge-camera-addon-', dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        aggregate_datasets(
            repo_ids=[base.name, prepared.name], roots=[base, prepared],
            aggr_repo_id=output.name, aggr_root=staging,
            # Rotate each source video into its own destination file. No video
            # concatenation/re-encoding or timestamp offset is introduced.
            video_files_size_in_mb=0.000001,
        )
        merged = json.loads((staging / 'meta/info.json').read_text())
        assert merged['total_frames'] == base_info['total_frames'] + info['total_frames']
        assert merged['total_episodes'] == base_info['total_episodes'] + info['total_episodes']
        frames = 0
        for file in sorted((staging / 'data').rglob('*.parquet')):
            table = pq.read_table(file)
            actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            assert actions.shape == (len(table), 6) and np.isfinite(actions).all()
            assert not any(key.startswith('observation.') and key != 'observation.images.wrist'
                           for key in table.column_names)
            frames += len(table)
        assert frames == merged['total_frames']
        provenance = {'sources': [str(base), str(addon)], 'source_info_sha256': original_hashes,
                      'frames': frames, 'episodes': merged['total_episodes'],
                      'input': ['observation.images.wrist'],
                      'action': ['dx', 'dy', 'dz', 'drx', 'dry', 'drz'],
                      'sampling': 'uniform over all merged frames; no source oversampling',
                      'videos': 'copied unchanged, not concatenated or re-encoded'}
        for source, digest in original_hashes.items():
            assert hashlib.sha256((Path(source) / 'meta/info.json').read_bytes()).hexdigest() == digest
        (staging / 'meta/merge_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        staging.rename(output)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
