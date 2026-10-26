"""Create a PI05 task-text view of mix4 without copying or changing recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

TASK = 'insert the ethernet cable into the hole'


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    info = json.loads((source/'meta/info.json').read_text())
    assert info['features']['observation.state']['names'] == ['ee_z']
    assert info['features']['action']['names'] == ['dx','dy','dz','drx','dry','drz']
    destination.mkdir(parents=True)
    try:
        shutil.copytree(source/'meta', destination/'meta')
        for name in ['data','videos']:
            (destination/name).symlink_to(source/name, target_is_directory=True)
        tasks = pq.read_table(source/'meta/tasks.parquet').to_pandas()
        assert len(tasks) == 1 and tasks['task_index'].iloc[0] == 0
        tasks.index = [TASK]
        tasks.index.name = 'task'
        tasks.to_parquet(destination/'meta/tasks.parquet')
        for path in (destination/'meta/episodes').rglob('*.parquet'):
            table = pq.read_table(path)
            idx = table.schema.get_field_index('tasks')
            table = table.set_column(idx,'tasks',pa.array([[TASK]]*len(table),type=table.schema.field('tasks').type))
            pq.write_table(table,path)
        assert (source/'meta/stats.json').read_bytes() == (destination/'meta/stats.json').read_bytes()
        provenance = dict(source=str(source),task=TASK,recordings='unchanged, linked to source',
            statistics='unchanged training-only statistics',
            split_sha256=hashlib.sha256((source/'meta/split.json').read_bytes()).hexdigest())
        (destination/'meta/pi05_preparation.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(json.dumps(provenance),flush=True)
    except Exception:
        shutil.rmtree(destination)
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('destination',type=Path)
    a=p.parse_args()
    prepare(a.source,a.destination)
