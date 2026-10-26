"""Merge approved fine datasets into wrist RGB + measured Z / action6, preserving videos."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def stats(values):
    values = values.astype(np.float64)
    result = {name: getattr(values, name)(axis=0).tolist() for name in ('min', 'max', 'mean', 'std')}
    result['count'] = [len(values)]
    for name, quantile in [('q01', .01), ('q10', .1), ('q50', .5), ('q90', .9), ('q99', .99)]:
        result[name] = np.quantile(values, quantile, axis=0).tolist()
    return result


def read_data(root):
    return pa.concat_tables([pq.read_table(file) for file in sorted((root / 'data').rglob('*.parquet'))]).sort_by('index')


def convert(source, output):
    info = json.loads((source / 'meta/info.json').read_text())
    table = read_data(source)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)[:, :6]
    raw = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    z = raw[:, [info['features']['observation.state']['names'].index('ee_z')]]
    ep = table['episode_index'].to_numpy()
    if len(table) != info['total_frames'] or not np.isfinite(actions).all() or not np.isfinite(z).all():
        raise ValueError(f'Invalid source data: {source}')
    np.testing.assert_array_equal(table['index'].to_numpy(), np.arange(len(table)))
    wanted = {'action', 'observation.state', 'observation.images.wrist', 'timestamp', 'frame_index',
              'episode_index', 'index', 'task_index'}

    def copy_file(src, dst):
        if Path(src).relative_to(source).parts[0] == 'videos':
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    shutil.copytree(source, output, copy_function=copy_file)
    for file in (output / 'data').rglob('*.parquet'):
        current = pq.read_table(file)
        indices = current['index'].to_numpy()
        current = current.drop([key for key in current.column_names if key not in wanted])
        for key, values in [('action', actions), ('observation.state', z)]:
            current = current.set_column(current.schema.get_field_index(key), key,
                pa.array(values[indices].tolist(), type=pa.list_(pa.float32(), values.shape[1])))
        pq.write_table(current.replace_schema_metadata(None), file)
    episode_stats = {int(e): stats(z[ep == e]) for e in np.unique(ep)}
    for file in (output / 'meta/episodes').rglob('*.parquet'):
        current = pq.read_table(file)
        current = current.drop([key for key in current.column_names if key.startswith('stats/')
                                and (key.split('/')[1] not in wanted or key.startswith('stats/observation.state/'))])
        for key in list(current.column_names):
            if key.startswith('stats/action/') and not key.endswith('/count'):
                current = current.set_column(current.schema.get_field_index(key), key,
                                             pa.array([row[:6] for row in current[key].to_pylist()]))
        for key in next(iter(episode_stats.values())):
            current = current.append_column('stats/observation.state/' + key,
                pa.array([episode_stats[e][key] for e in current['episode_index'].to_pylist()]))
        pq.write_table(current.replace_schema_metadata(None), file)
    info['features'] = {key: deepcopy(value) for key, value in info['features'].items() if key in wanted}
    info['features']['action'].update(shape=[6], names=['dx', 'dy', 'dz', 'drx', 'dry', 'drz'])
    info['features']['observation.state'].update(shape=[1], names=['ee_z'])
    (output / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')
    source_stats = json.loads((source / 'meta/stats.json').read_text())
    source_stats = {key: value for key, value in source_stats.items() if key in wanted}
    source_stats.update(action=stats(actions), **{'observation.state': stats(z)})
    (output / 'meta/stats.json').write_text(json.dumps(source_stats, indent=2) + '\n')
    return actions, z, table


def prepare(plan_path, split_path, output):
    plan, split = [json.loads(path.read_text()) for path in (plan_path, split_path)]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.fine-mix4-', dir=output.parent) as temp:
        temporary = Path(temp)
        prepared, labels, states, episodes, frames, timestamps, videos = [], [], [], [], [], [], []
        offset = 0
        for item in plan['sources']:
            source = Path(item['root'])
            if digest(source / 'meta/info.json') != item['info_sha256']:
                raise ValueError(f'Source metadata changed since approval: {source}')
            destination = temporary / item['name']
            action, z, table = convert(source, destination)
            prepared.append(destination)
            labels.append(action)
            states.append(z)
            episodes.append(table['episode_index'].to_numpy() + offset)
            frames.append(table['frame_index'].to_numpy())
            timestamps.append(table['timestamp'].to_numpy())
            videos.extend(digest(p) for p in sorted((source / 'videos').rglob('*.mp4')))
            offset += item['episodes']
            print('Prepared',item['name'],len(table),'frames',flush=True)
        staging = temporary / output.name
        aggregate_datasets(repo_ids=[p.name for p in prepared], roots=prepared,
                           aggr_repo_id=output.name, aggr_root=staging, video_files_size_in_mb=0.000001)
        actions, z, ep = np.concatenate(labels), np.concatenate(states), np.concatenate(episodes)
        merged = read_data(staging)
        np.testing.assert_array_equal(np.asarray(merged['action'].to_pylist(), dtype=np.float32), actions)
        np.testing.assert_array_equal(np.asarray(merged['observation.state'].to_pylist(), dtype=np.float32), z)
        np.testing.assert_array_equal(merged['episode_index'].to_numpy(), ep)
        np.testing.assert_array_equal(merged['frame_index'].to_numpy(), np.concatenate(frames))
        np.testing.assert_array_equal(merged['timestamp'].to_numpy(), np.concatenate(timestamps))
        np.testing.assert_array_equal(merged['index'].to_numpy(), np.arange(len(merged)))
        assert Counter(videos) == Counter(digest(p) for p in (staging / 'videos').rglob('*.mp4'))
        train, val = set(split['train_episodes']), set(split['validation_episodes'])
        assert not train & val and train | val == set(range(offset))
        selected = np.isin(ep, list(train))
        assert selected.sum() == split['train_frames'] and (~selected).sum() == split['validation_frames']
        stat_path = staging / 'meta/stats.json'
        all_stats = json.loads(stat_path.read_text())
        all_stats.update(action=stats(actions[selected]), **{'observation.state': stats(z[selected])})
        stat_path.write_text(json.dumps(all_stats, indent=2) + '\n')
        shutil.copy2(split_path, staging / 'meta/split.json')
        provenance = dict(sources=plan['sources'], frames=len(merged), episodes=offset,
            train_frames=int(selected.sum()), validation_frames=int((~selected).sum()),
            normalization='state/action statistics from training episodes only; ImageNet visual statistics',
            verification='all action/Z/index/timestamp rows and original video SHA256 verified',
            task='uniform frame sampling from training episodes; no source oversampling')
        (staging / 'meta/merge_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        staging.rename(output)
    print('VERIFIED',output,json.dumps(provenance),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.plan,args.split,args.output)
