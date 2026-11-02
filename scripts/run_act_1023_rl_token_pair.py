"""Run the approved two RL Token jobs sequentially, waiting for an idle GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def now():return datetime.datetime.now().astimezone().isoformat()


def idle_gpu():
    row=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True).strip()
    used,util=map(lambda x:int(x.strip()),row.split(','))
    apps=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    return used<1024 and util<=5 and not apps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True)
    a=p.parse_args();root=a.project.resolve()
    configs=root/'configs/act_1023_z_rl_token_207'
    names=['act_1023_z_token','act_1023_z_cvae_token']
    record=dict(host='10.0.40.207',launcher_pid=os.getpid(),started_at=now(),
        execution='sequential',gpu=0,runs=[dict(name=n,status='QUEUED') for n in names])
    info=configs/'run_info.json'
    def save():
        temp=info.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(info)
    save()
    entry=('from act_rlt.dataset_compat import install_scalar_state_dataset_support; '
           'install_scalar_state_dataset_support(); from act_rlt import register; register(); '
           'from lerobot.scripts.lerobot_train import main; main()')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=f'{root}/src:{root}',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
        HF_DATASETS_CACHE=str(root/'.cache/hf_datasets'),TORCH_HOME=str(root/'.cache/torch'))
    (root/'logs/vga_token').mkdir(parents=True,exist_ok=True)
    for job in record['runs']:
        name=job['name'];cfg_path=configs/(name+'.json');cfg=json.loads(cfg_path.read_text())
        output=Path(cfg['output_dir']);assert not output.exists(),f'Output already exists: {output}'
        job.update(status='WAITING_GPU');save()
        while not idle_gpu():time.sleep(15)
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log=root/'logs/vga_token'/f'{name}_{stamp}.log'
        cmd=[str(a.python),'-u','-c',entry,f'--config_path={cfg_path}']
        with log.open('ab',buffering=0) as f:
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            job.update(status='RUNNING',pid=proc.pid,started_at=now(),log=str(log),
                       output_dir=str(output),config=str(cfg_path),command=cmd);save()
            code=proc.wait()
        job.update(exit_code=code,finished_at=now())
        state=output/'checkpoints/last/training_state/training_step.json'
        if code!=0 or not state.exists() or json.loads(state.read_text())['step']!=cfg['steps']:
            job['status']='FAILED';save();raise RuntimeError(f'{name} failed; inspect {log}')
        job.update(status='COMPLETE',step=cfg['steps']);save()
    record.update(status='COMPLETE',finished_at=now());save()


if __name__=='__main__':main()
