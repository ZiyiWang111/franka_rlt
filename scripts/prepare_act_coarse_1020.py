"""Prepare wrist-only ACT joint-position and TCP-delta datasets without changing source."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def stats(values):
    values = values.astype(np.float64)
    result = {k: getattr(values, k)(axis=0).tolist() for k in ('min', 'max', 'mean', 'std')}
    result['count'] = [len(values)]
    for k, q in [('q01', .01), ('q10', .1), ('q50', .5), ('q90', .9), ('q99', .99)]:
        result[k] = np.quantile(values, q, axis=0).tolist()
    return result


def prepare(source, output, variant):
    if output.exists():
        raise FileExistsError(output)
    info = json.loads((source / 'meta/info.json').read_text())
    files = sorted((source / 'data').rglob('*.parquet'))
    table = pa.concat_tables([pq.read_table(p) for p in files])
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    episode = np.asarray(table['episode_index'].to_pylist())
    frame = np.asarray(table['frame_index'].to_pylist())
    indices = np.asarray(table['index'].to_pylist())
    np.testing.assert_array_equal(indices, np.arange(len(table)))
    names = info['features']['observation.state']['names']
    joint_indices = [names.index(f'joint_{i}') for i in range(7)]
    if variant == 'rgb_joint_abs_joint':
        inputs = state[:, joint_indices]
        labels = inputs.copy()
        same = episode[:-1] == episode[1:]
        assert np.all((frame[1:] - frame[:-1])[same] == 1)
        labels[np.flatnonzero(same)] = inputs[np.flatnonzero(same) + 1]
        state_names = [f'joint_{i}' for i in range(7)]
        action_names = state_names
    else:
        inputs = state[:, [names.index('ee_z')]]
        labels = np.asarray(table['action'].to_pylist(), dtype=np.float32)[:, :6]
        state_names = ['ee_z']
        action_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
    assert np.isfinite(inputs).all() and np.isfinite(labels).all()
    wanted = {'action', 'observation.state', 'observation.images.wrist', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    features = {k: deepcopy(v) for k, v in info['features'].items() if k in wanted}
    features['action'].update(shape=[labels.shape[1]], names=action_names)
    features['observation.state'].update(shape=[inputs.shape[1]], names=state_names)
    output.mkdir(parents=True)
    shutil.copytree(source / 'meta', output / 'meta')
    video_source = source / 'videos/observation.images.wrist'
    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    shutil.copytree(video_source, output / 'videos/observation.images.wrist', copy_function=link_or_copy)
    offset = 0
    for f in files:
        t = pq.read_table(f)
        t = t.drop([k for k in t.column_names if k not in features])
        for key, values in [('observation.state', inputs), ('action', labels)]:
            arr = pa.array(values[offset:offset + len(t)].tolist(), type=pa.list_(pa.float32(), values.shape[1]))
            t = t.set_column(t.schema.get_field_index(key), key, arr)
        dest = output / f.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t.replace_schema_metadata(None), dest)
        offset += len(t)
    updated = {'observation.state': inputs, 'action': labels}
    per_episode = {int(e): {k: stats(v[episode == e]) for k, v in updated.items()} for e in np.unique(episode)}
    for f in (output / 'meta/episodes').rglob('*.parquet'):
        t = pq.read_table(f)
        drops = [k for k in t.column_names if k.startswith('stats/') and (k.split('/')[1] not in features or k.split('/')[1] in updated)]
        t = t.drop(drops)
        for feature in updated:
            for stat in stats(updated[feature]):
                t = t.append_column(f'stats/{feature}/{stat}', pa.array([per_episode[e][feature][stat] for e in t['episode_index'].to_pylist()]))
        pq.write_table(t.replace_schema_metadata(None), f)
    all_stats = json.loads((source / 'meta/stats.json').read_text())
    all_stats = {k: v for k, v in all_stats.items() if k in features}
    all_stats.update({k: stats(v) for k, v in updated.items()})
    (output / 'meta/stats.json').write_text(json.dumps(all_stats, indent=2) + '\n')
    info['features'] = features
    (output / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')
    (output / 'meta/training_conversion.json').write_text(json.dumps(dict(source=str(source.resolve()), source_info_sha256=hashlib.sha256((source/'meta/info.json').read_bytes()).hexdigest(), variant=variant, final_frame='hold joint positions / zero TCP delta', frames=len(table), episodes=len(per_episode)), indent=2)+'\n')
    # Verify stored labels and episode alignment after writing.
    written = pa.concat_tables([pq.read_table(p) for p in sorted((output/'data').rglob('*.parquet'))])
    np.testing.assert_array_equal(np.asarray(written['action'].to_pylist(), dtype=np.float32), labels)
    np.testing.assert_array_equal(np.asarray(written['observation.state'].to_pylist(), dtype=np.float32), inputs)
    print(f'{variant}: verified {len(table)} frames / {len(per_episode)} episodes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    for variant in ('rgb_joint_abs_joint', 'rgb_z_delta_tcp'):
        prepare(args.source, args.source.parent / f'act_coarse_1020_{variant}', variant)
