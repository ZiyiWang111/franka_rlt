"""Restore measured joints and EE pose to a hold-tail dataset; preserve RGB/actions."""
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

from scripts.prepare_act_coarse_1020 import stats


STATE_NAMES = [f'joint_{i}' for i in range(7)] + [f'ee_{axis}' for axis in ('x', 'y', 'z', 'rx', 'ry', 'rz')]


def read_data(root):
    return pa.concat_tables([pq.read_table(path) for path in sorted((root / 'data').rglob('*.parquet'))]).sort_by('index')


def prepare(hold, original, output):
    hold, original, output = [path.resolve() for path in (hold, original, output)]
    if output.exists() or output in (hold, original) or hold in output.parents or original in output.parents:
        raise ValueError('Output must be a new independent dataset')
    info = json.loads((hold / 'meta/info.json').read_text())
    raw_info = json.loads((original / 'meta/info.json').read_text())
    tail = json.loads((hold / 'meta/static_tail.json').read_text())
    if info['fps'] != raw_info['fps'] or tail['tail_frames'] != info['fps']:
        raise ValueError('Expected a one-second static tail at the original frame rate')
    if info['features']['observation.state']['names'] != ['ee_z']:
        raise ValueError('Expected scalar measured TCP Z as the hold dataset state')
    raw = read_data(original)
    data = read_data(hold)
    np.testing.assert_array_equal(data['index'].to_numpy(), np.arange(len(data)))
    np.testing.assert_array_equal(raw['index'].to_numpy(), np.arange(len(raw)))
    raw_ep, hold_ep = raw['episode_index'].to_numpy(), data['episode_index'].to_numpy()
    np.testing.assert_array_equal(np.unique(raw_ep), np.unique(hold_ep))
    raw_state = np.asarray(raw['observation.state'].to_pylist(), dtype=np.float32)
    old_state = np.asarray(data['observation.state'].to_pylist(), dtype=np.float32)
    raw_action = np.asarray(raw['action'].to_pylist(), dtype=np.float32)
    actions = np.asarray(data['action'].to_pylist(), dtype=np.float32)
    names = raw_info['features']['observation.state']['names']
    columns = [names.index(name) for name in STATE_NAMES]
    restored = np.empty((len(data), len(STATE_NAMES)), dtype=np.float32)
    per_episode = {}
    for episode in np.unique(raw_ep):
        source_indices = np.flatnonzero(raw_ep == episode)
        destination_indices = np.flatnonzero(hold_ep == episode)
        length = len(source_indices)
        if len(destination_indices) != length + tail['tail_frames']:
            raise ValueError(f'Episode {episode}: unexpected static tail length')
        np.testing.assert_array_equal(data['frame_index'].to_numpy()[destination_indices], np.arange(len(destination_indices)))
        np.testing.assert_array_equal(raw['frame_index'].to_numpy()[source_indices], np.arange(length))
        for key in ('timestamp', 'task_index'):
            np.testing.assert_array_equal(data[key].to_numpy()[destination_indices[:length]], raw[key].to_numpy()[source_indices])
        np.testing.assert_array_equal(actions[destination_indices[:length]], raw_action[source_indices, :6])
        np.testing.assert_array_equal(actions[destination_indices[length:]], np.zeros((tail['tail_frames'], 6)))
        indices = np.concatenate([source_indices, np.repeat(source_indices[-1], tail['tail_frames'])])
        np.testing.assert_array_equal(old_state[destination_indices, 0], raw_state[indices, names.index('ee_z')])
        restored[destination_indices] = raw_state[indices][:, columns]
        per_episode[int(episode)] = stats(restored[destination_indices])
    if len(data) != info['total_frames'] or len(per_episode) != info['total_episodes'] or not np.isfinite(restored).all():
        raise ValueError('Invalid restored states or inconsistent dataset counts')

    def copy_file(src, dst):
        if Path(src).relative_to(hold).parts[0] == 'videos':
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.coarse-state13-', dir=output.parent) as temp:
        staging = Path(temp) / output.name
        shutil.copytree(hold, staging, copy_function=copy_file)
        for file in sorted((staging / 'data').rglob('*.parquet')):
            table = pq.read_table(file)
            indices = table['index'].to_numpy()
            table = table.set_column(table.schema.get_field_index('observation.state'), 'observation.state',
                                     pa.array(restored[indices].tolist(), type=pa.list_(pa.float32(), 13)))
            pq.write_table(table.replace_schema_metadata(None), file)
        for file in (staging / 'meta/episodes').rglob('*.parquet'):
            table = pq.read_table(file)
            table = table.drop([name for name in table.column_names if name.startswith('stats/observation.state/')])
            for key in next(iter(per_episode.values())):
                values = [per_episode[e][key] for e in table['episode_index'].to_pylist()]
                table = table.append_column('stats/observation.state/' + key, pa.array(values))
            pq.write_table(table.replace_schema_metadata(None), file)
        info['features']['observation.state'].update(shape=[13], names=STATE_NAMES)
        (staging / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')
        stat_path = staging / 'meta/stats.json'
        dataset_stats = json.loads(stat_path.read_text())
        dataset_stats['observation.state'] = stats(restored)
        stat_path.write_text(json.dumps(dataset_stats, indent=2) + '\n')
        provenance = dict(hold_source=str(hold), full_state_source=str(original), state_names=STATE_NAMES,
                          hold_info_sha256=hashlib.sha256((hold/'meta/info.json').read_bytes()).hexdigest(),
                          original_info_sha256=hashlib.sha256((original/'meta/info.json').read_bytes()).hexdigest(),
                          frames=len(data), episodes=len(per_episode), tail_frames=tail['tail_frames'],
                          verified_alignment=['episode_index', 'frame_index', 'timestamp', 'task_index', 'ee_z', 'action'],
                          state_units='joint angles radians; base-frame EE XYZ metres and rotation vector radians')
        (staging / 'meta/full_state_recovery.json').write_text(json.dumps(provenance, indent=2) + '\n')
        written = read_data(staging)
        np.testing.assert_array_equal(np.asarray(written['observation.state'].to_pylist(), dtype=np.float32), restored)
        for key in data.column_names:
            if key != 'observation.state':
                assert written[key].equals(data[key]), f'Unexpected change to {key}'
        staging.rename(output)
    print(f'VERIFIED {output}: {len(data)} frames / {len(per_episode)} episodes, measured 13D state, unchanged RGB/actions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hold', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.hold, args.original, args.output)
