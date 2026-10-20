#!/usr/bin/env python3
"""Restore only measured TCP Z to a merged camera/action6 dataset, frame by frame."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def columns(root):
    tables = [pq.read_table(p) for p in sorted((root / 'data').rglob('*.parquet'))]
    return {key: np.concatenate([np.asarray(t[key].to_pylist()) for t in tables])
            for key in ('action', 'observation.state', 'episode_index', 'frame_index', 'timestamp', 'index')}


def stats(z):
    values = z.astype(np.float64)
    result = {key: [float(value)] for key, value in (
        ('min', values.min()), ('max', values.max()), ('mean', values.mean()), ('std', values.std()))}
    result['count'] = [len(z)]
    for name, quantile in [('q01', .01), ('q10', .1), ('q50', .5), ('q90', .9), ('q99', .99)]:
        result[name] = [float(np.quantile(values, quantile))]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True,
                        help='Original datasets, in the exact merged order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merged, output = args.merged.resolve(), args.output.resolve()
    sources = [p.resolve() for p in args.sources]
    if output.exists() or output == merged or merged in output.parents:
        raise ValueError('Output must be a new independent dataset')
    info = json.loads((merged / 'meta/info.json').read_text())
    assert 'observation.state' not in info['features']
    expected = {key: [] for key in ('action', 'z', 'episode_index', 'frame_index', 'timestamp', 'index')}
    ep_offset = frame_offset = 0
    source_hashes = {}
    for source in sources:
        raw_info = (source / 'meta/info.json').read_bytes()
        source_hashes[str(source)] = hashlib.sha256(raw_info).hexdigest()
        metadata = json.loads(raw_info)
        names = metadata['features']['observation.state']['names']
        z_index = names.index('ee_z')
        data = columns(source)
        assert len(data['index']) == metadata['total_frames']
        expected['action'].append(data['action'][:, :6])
        expected['z'].append(data['observation.state'][:, z_index].astype(np.float32))
        for key in ('frame_index', 'timestamp'):
            expected[key].append(data[key])
        expected['episode_index'].append(data['episode_index'] + ep_offset)
        expected['index'].append(data['index'] + frame_offset)
        ep_offset += metadata['total_episodes']
        frame_offset += metadata['total_frames']
    expected = {key: np.concatenate(values) for key, values in expected.items()}
    assert frame_offset == info['total_frames'] and ep_offset == info['total_episodes']
    assert np.isfinite(expected['z']).all() and expected['z'].std() > 0
    episode_stats = {int(ep): stats(expected['z'][expected['episode_index'] == ep])
                     for ep in np.unique(expected['episode_index'])}
    output.parent.mkdir(parents=True, exist_ok=True)

    def copy_file(src, dst):
        if Path(src).relative_to(merged).parts[0] == 'videos':
            try:
                return os.link(src, dst)
            except OSError:
                pass
        return shutil.copy2(src, dst)

    with TemporaryDirectory(prefix='.act-tcp-z-', dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        shutil.copytree(merged, staging, copy_function=copy_file)
        for file in sorted((staging / 'data').rglob('*.parquet')):
            table = pq.read_table(file)
            indices = np.asarray(table['index'].to_pylist(), dtype=np.int64)
            for key in ('index', 'episode_index', 'frame_index', 'timestamp', 'action'):
                np.testing.assert_array_equal(np.asarray(table[key].to_pylist()), expected[key][indices])
            table = table.append_column('observation.state', pa.array(
                expected['z'][indices, None].tolist(), type=pa.list_(pa.float32(), 1)))
            pq.write_table(table.replace_schema_metadata(None), file)
        for file in (staging / 'meta/episodes').rglob('*.parquet'):
            table = pq.read_table(file)
            episodes = table['episode_index'].to_pylist()
            for key in next(iter(episode_stats.values())):
                table = table.append_column(f'stats/observation.state/{key}',
                    pa.array([episode_stats[ep][key] for ep in episodes]))
            pq.write_table(table.replace_schema_metadata(None), file)
        info['features']['observation.state'] = {'dtype': 'float32', 'shape': [1], 'names': ['ee_z']}
        (staging / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')
        stats_file = staging / 'meta/stats.json'
        all_stats = json.loads(stats_file.read_text())
        all_stats['observation.state'] = stats(expected['z'])
        stats_file.write_text(json.dumps(all_stats, indent=2) + '\n')
        provenance = dict(merged=str(merged), sources=[str(p) for p in sources],
            source_info_sha256=source_hashes, state_names=['ee_z'], state_units='metres',
            state_semantics='measured current absolute TCP Z in base frame',
            frames=frame_offset, episodes=ep_offset, z_stats=all_stats['observation.state'],
            verified_alignment=['index', 'episode_index', 'frame_index', 'timestamp', 'action[:6]'])
        (staging / 'meta/tcp_z_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        for source, digest in source_hashes.items():
            assert hashlib.sha256((Path(source) / 'meta/info.json').read_bytes()).hexdigest() == digest
        staging.rename(output)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
