import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from .generator import DEFAULT_CONFIG, ROOT, layout_config
from .maps import TASKS, build_map, convert_reference


def main():
    p=argparse.ArgumentParser(description='Generate and validate offline forest maps')
    sub=p.add_subparsers(dest='command',required=True)
    executor=sub.add_parser('generate-executor-pilot',help='Executor diagnostics plus native forest previews with separate readiness labels')
    executor.add_argument('--output',default=str(ROOT.parent/'outputs/forest_vln/executor_pilot/maps'))
    executor.add_argument('--seeds',nargs='+',type=int,default=[202610240,202610241,202610242])
    suite=sub.add_parser('generate-executor-suite',help='Freeze 40 development tasks before flight')
    suite.add_argument('--output',required=True)
    suite.add_argument('--source-maps',default='outputs/forest_vln/executor_pilot/maps')
    from scripts.generate_forest_multirow_dataset import add_arguments,main as multirow_main
    multi=sub.add_parser('generate-multirow',help='Recommended: native multirow inspection maps with five obstacle compositions')
    add_arguments(multi)
    g=sub.add_parser('generate');g.add_argument('--config',default=str(DEFAULT_CONFIG))
    g.add_argument('--output',default=str(ROOT.parent/'outputs/forest/maps'))
    g.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    g.add_argument('--seeds',nargs='+',type=int,default=[0])
    g.add_argument('--split',choices=['train','val','test'],default='train',help='assign every requested seed to this dataset split (default: train)')
    g.add_argument('--difficulty',choices=['easy','medium','hard'],default='easy')
    g.add_argument('--variant',choices=['center','offset','approach'],default='center')
    g.add_argument('--layout',choices=['long','short'],default='short',help='short: 36 m routes, rolling terrain, wooded borders and 2–3 obstacles')
    g.add_argument('--flight-config',default=str(ROOT.parent/'cfg/task/ForestSurvey.yaml'))
    r=sub.add_parser('convert');r.add_argument('source');r.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='generate-executor-suite':
        from .executor_suite import build_executor_suite
        print(build_executor_suite(args.output,args.source_maps))
        return
    if args.command=='generate-executor-pilot':
        from .executor_pilot import build_executor_maps
        build_executor_maps(args.output,args.seeds);return
    if args.command=='generate-multirow':
        multirow_main(args=args);return
    if args.command=='convert':
        print(convert_reference(args.source,args.output));return
    config=layout_config(json.loads(Path(args.config).read_text()),args.layout)
    config['flight_height']=max(2.5,config['flight_height'])
    seeds=args.seeds
    print(f'Generating {len(seeds)} requested seeds; assigning all to {args.split}.',flush=True)
    flight=OmegaConf.load(args.flight_config).flight
    results=[]
    for seed in seeds:
        for task in args.tasks:
            try:
                folder=build_map(args.output,seed,config,task,flight,args.difficulty,args.variant,split=args.split)
                results.append(dict(seed=seed,task=task,reverse=False,split=args.split,passed=True,path=str(folder)))
            except ValueError as error:
                results.append(dict(seed=seed,task=task,reverse=False,split=args.split,passed=False,reason=str(error)))
                print(f'{seed}/{task}: {error}',flush=True)
    split_suffix=f'_{args.split}'
    report=Path(args.output)/args.difficulty/f'generation_{"_".join(map(str,seeds))}_{"_".join(args.tasks)}_{args.variant}{split_suffix}.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(f'Report: {report}',flush=True)
    if not all(r['passed'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    main()
