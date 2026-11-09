"""下载主实验模型到项目 models/；不修改仿真环境。"""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',nargs='+',choices=['2B','4B'],default=['2B','4B'])
    p.add_argument('--output',default='models')
    args=p.parse_args()
    for size in args.sizes:
        name='Qwen3.5-'+size
        snapshot_download('Qwen/'+name,local_dir=Path(args.output)/name,max_workers=2,
            allow_patterns=['*.json','*.jinja','*.safetensors','*.txt','*.model','LICENSE','README.md'])
        directory=Path(args.output)/name
        revision=(directory/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]
        (directory/'model_origin.json').write_text(json.dumps(dict(repo_id='Qwen/'+name,revision=revision),indent=2)+'\n')
        print('Downloaded '+name,flush=True)
