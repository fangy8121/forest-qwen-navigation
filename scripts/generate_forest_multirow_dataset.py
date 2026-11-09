"""Generate native multirow maps with frozen independent splits; retain rejected candidate records."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time
import tempfile
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from forest_generator.multirow_tasks import PROFILES
from forest_generator.multirow_layout import layout_options
from forest_generator.clearance_profiles import PROFILE_NAMES,clearance_profile

LABELS=dict(light='少量绕障',balanced='单侧与双侧混合',sequences='连续绕障',
            mixed_bypass='多种绕障组合',neighbor_detour='封堵后借邻行绕行')

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def records(counts=(20,5,5),seed_start=20261700,profiles=PROFILES,seeds=None,split='train'):
    if not profiles or any(p not in PROFILES for p in profiles) or len(set(profiles))!=len(profiles):
        raise ValueError('Choose distinct supported profiles')
    if len(counts)!=3 or any(n<0 for n in counts) or not sum(counts):
        raise ValueError('counts must contain three nonnegative integers with a positive total')
    if seeds is not None and (not seeds or len(set(seeds))!=len(seeds)):
        raise ValueError('seeds must be nonempty and unique')
    if seed_start<0 or (seeds is not None and any(s<0 for s in seeds)):
        raise ValueError('Seeds must be nonnegative')
    if split not in ('train','val','test'):raise ValueError('Unknown split')
    entries=[]
    offset=0
    groups=[(split,len(seeds))] if seeds is not None else list(zip(('train','val','test'),counts))
    for split,count in groups:
        for i in range(count):
            entries.append(dict(path=f'{split}/{i:03d}',split=split,seed=seeds[i] if seeds is not None else seed_start+offset+i,
                                profile=profiles[i%len(profiles)]))
        offset+=count
    return entries

def like_records(maps):
    """Reuse the exact path/split/seed/profile of an existing maps directory."""
    entries=[]
    for path in sorted(Path(maps).glob('*/*/map.json')):
        info=json.loads(path.read_text())
        entries.append(dict(path=str(path.parent.relative_to(maps)),split=info['split'],
                            seed=info['parameters']['seed'],profile=info['tasks']['profile']))
    if not entries:raise ValueError(f'No maps under {maps}')
    if len({e['seed'] for e in entries})!=len(entries):raise ValueError('Duplicate seeds in --like source')
    return entries

def worker(root,entry,complete_reference_audit=False,max_attempts=3,start_attempt=0,layout=None,clearance='legacy'):
    from scripts.build_forest_multirow_native import build
    path=Path(root)/entry['path'];start=time.monotonic()
    # Each worker has its own log to avoid interleaved multiline errors; merged
    # into the single public log by the parent, then removed.
    descriptor,name=tempfile.mkstemp(prefix='forest-dataset-',suffix='.log',dir='/tmp')
    os.close(descriptor);scratch=Path(name)
    reviews=[]
    try:
        with scratch.open('w') as stream,redirect_stdout(stream),redirect_stderr(stream):
            if complete_reference_audit:raise ValueError('Formal dataset publishes PASS only; old audit mode is historical')
            assert not path.exists(), 'Never replace a published map'
            info=None
            for attempt in range(start_attempt,start_attempt+max_attempts):
                with tempfile.TemporaryDirectory(prefix='forest-candidate-',dir='/tmp') as temp:
                    candidate=Path(temp)/'map'
                    try:
                        info=build(candidate,entry['seed'],split=entry['split'],task_profile=entry['profile'],
                                   flat=True,mesh_preview=False,continue_after_task_mismatch=True,task_attempt=attempt,
                                   layout=layout,clearance=clearance)
                    except ValueError as error:
                        # Task placement rejects before map.json exists: try the next
                        # placement attempt with the same forest seed.
                        print(f'attempt {attempt}: {error}',flush=True)
                        reviews.append(dict(map=entry['path'],attempt=attempt,
                                            info=dict(validation=dict(passed=False,reason=str(error)))))
                        continue
                    if info['validation']['passed']:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        shutil.move(str(candidate),str(path))
                        break
                    reviews.append(dict(map=entry['path'],attempt=attempt,info=info))
        if info is None:raise ValueError('every attempt failed task placement: '+reviews[-1]['info']['validation']['reason'])
        result=dict(entry,status='PASS' if info['validation']['passed'] else 'NEEDS_REPAIR',
                    validation=info['validation'],geometry_id=info['geometry_id'],
                    original_trees=info['provenance']['source_tree_count'],
                    added_trees=info['tasks']['inserted_trees'],aisles=len(info['lane_centers_y']),
                    task_assignments=info['tasks']['assignments'],events=info['tasks']['events'],
                    stem_placement_audit=info['tasks'].get('stem_placement_audit'),
                    branch_placement_audit=info['tasks'].get('branch_placement_audit'),
                    omitted_redundant_stems=len(info['tasks'].get('omitted_redundant_stems',[])),
                    accepted_attempt=attempt if info['validation']['passed'] else None,
                    last_attempt=attempt,
                    candidate_reviews=[dict(attempt=r['attempt'],reason=r['info']['validation'].get('reason')) for r in reviews])
    except Exception as error:
        import traceback
        with scratch.open('a') as stream:traceback.print_exc(file=stream)
        result=dict(entry,status='NEEDS_REPAIR',validation=dict(passed=False,reason=repr(error)))
    result['seconds']=round(time.monotonic()-start,2)
    log=scratch.read_text();scratch.unlink()
    return result,log,reviews

def render(root,manifest):
    results=manifest.get('results',[])
    summary={s:dict(Counter(r['status'] for r in results if r['split']==s)) for s in ('train','val','test')}
    references={s:sum(r['validation'].get('continuous_reference_passed',r['validation']['passed'])
                       for r in results if r['split']==s) for s in summary}
    manifest['summary']=summary
    manifest['continuous_references']=references
    write(root/'manifest.json',manifest)
    # The simulator's automatic scene selection consumes index.json.
    index=[]
    for result in results:
        if result['status']!='PASS':continue
        info=json.loads((root/result['path']/'map.json').read_text())
        index.append(dict(map_id=result['path'],scene_id=info['scene_id'],task=info['task'],split=result['split']))
    write(root/'index.json',index)
    accepted=sum(r['status']=='PASS' for r in results)
    targets=Counter(e['split'] for e in manifest['maps'])
    target_text=f'训练 {targets["train"]} / 验证 {targets["val"]} / 测试 {targets["test"]}'
    layout=manifest.get('layout',dict(mode='legacy'))
    clearance=manifest.get('clearance_profile',dict(name='legacy',clearance_m=.75,horizontal_m=None))
    clearance_text=(f'{clearance["clearance_m"]:g} m（{clearance["name"]}，按导航包络推导；另以水平净空 '
                    f'{clearance["horizontal_m"]:g} m 做最终否决）' if clearance.get('horizontal_m') else
                    f'{clearance["clearance_m"]:g} m')
    layout_text=('沿用原生树木模板、枝干分布和起伏地形；每行名义长度 '
                 f'{layout["length_range"][0]:g}–{layout["length_range"][1]:g} m，起点错位范围 '
                 f'{2*layout["start_jitter"]:g} m。通道端点由实际树干和枝干边界决定，换行采用连续曲线。'
                 if layout['mode']=='variable' else
                 '60×36 m 场地中放置原生成器采样的 40×32 m 林分，平移 (10,3) m；保留超出名义行长的末棵树。')
    audit_link=(' · <a href="verification.json">导出文件复核</a>' if (root/'verification.json').exists() else '')
    timing_note=('完整导出复核已通过：重新核对原生几何、三维净空、运动限制和参考阶段。'
                 if manifest.get('artifact_verification',{}).get('passed') else '')
    cards=[]
    for split in ('train','val','test'):
        cards.append(f'<h2>{split} · {summary[split]} · 完整安全参考 {references[split]} 张</h2><div class="grid">')
        bypath={r['path']:r for r in results}
        for entry in manifest['maps']:
            if entry['split']!=split:continue
            r=bypath.get(entry['path'],dict(entry,status='待生成',validation={}))
            p=r['path'];v=r['validation'];reason=html.escape(str(v.get('reason','')))
            image=f'<a href="{p}/overview.png"><img src="{p}/overview.png"></a>' if (root/p/'overview.png').exists() else ''
            route=(f'参考全长 {v["route_length"]:.1f} m；{reason}' if 'route_length' in v else reason)
            links=(f'<a href="{p}/map.json">参数与检查</a> · <a href="{p}/scene.usd">场景 USD</a>'
                   if (root/p/'map.json').exists() else '<span>尚未发布合格地图</span>')
            if (root/p/'03_实际网格视图.png').exists():links+=f' · <a href="{p}/03_实际网格视图.png">三维网格</a>'
            if (root/p/'terrain_route.png').exists():links+=f' · <a href="{p}/terrain_route.png">地形与航高</a>'
            cards.append(f'<article><h3>{p} · {r["status"]}</h3><p>{LABELS[r["profile"]]} · seed {r["seed"]}</p>'
                         f'{image}<p>{route}</p>{links}</article>')
        cards.append('</div>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>新版多行地图数据集</title>'
        '<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;padding:12px;background:#f5f6f7}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}'
        'article{background:white;padding:16px;border-radius:8px}img{width:100%}p{line-height:1.6}</style>'
        f'<h1>多行地图：已验收 {accepted}/{len(manifest["maps"])} 张</h1><p>目标：{target_text}；以下数量只统计已通过地图验收的场景。</p>'
        f'<p>{layout_text}橙色为额外任务障碍，绿色为原生树干，棕色为所有高度枝干投影，蓝色为通过检查的完整参考路线。俯视重叠不等于三维碰撞。</p>'
        '<p>修正版：保留原生树林；新增障碍使用交错的不规则枯立木簇，按原生三维网格选择局部通路位置。每图计划 4–9 个障碍事件，保留至少两个自然通道。'
        '新增树干与其他树干至少间隔 0.15 m，与原生枝干保留至少 0.05 m 的保守间隔；枯立木高度随放置条件变化。已有树干占据的位置不重复插树。'
        f'这个间距是造景防穿插规则，与飞行中心净空 {clearance_text} 是不同的量。枝干仍可能在俯视图上交叠。</p>'
        '<p><b>PASS 仅指完整参考曲线通过几何、运动约束和任务门检查；尚无实际扫描、滚动六点执行或 Qwen 成绩。</b>'
        '正式目录只发布 PASS 地图。失败候选的完整树木参数与原因集中记录在 candidate_reviews.jsonl，未计入合格数量。测试集森林种子预先固定，仅做地图设计验收，不按模型表现筛选。</p>'
        f'<p>{timing_note}</p><p><a href="README.md">说明</a> · <a href="manifest.json">固定划分与审核清单</a>{audit_link}</p>'+''.join(cards))
    (root/'README.md').write_text('# 新版多行地图\n\n直接打开 [总览](index.html)。\n\n'
        f'预先固定 {len(manifest["maps"])} 个独立种子：{target_text}。各图的种子与任务组合见 manifest.json；组合按所选顺序循环分配，一个种子对应一张地图。'
        '这是第一批可行性地图，不足以证明对真实未知森林的泛化。\n\n'
        f'{layout_text}'
        '行距、株距、枝干与地形随种子变化。每张至少两个通道不增加任务障碍；其余随机分配任务。'
        '额外障碍为使用原树干模板的不规则枯立木簇，原生枝干全部保留；尚不包含倒木、灌丛和动态物体。'
        '每图计划 4–9 个障碍事件，连续绕障通道放三个事件；其余任务通道通常两个，整行封堵仅一个。'
        '位置选择在既定邻域内检查原生三维网格的局部地形跟随通路，记录失败尝试，不更换森林种子。\n\n'
        '修正版保留相同种子与划分，新增树干之间及其与原生树干之间至少保留 0.15 m 表面间隔；'
        '原树干占据的位置不重复插入。省略位置记录在 tasks.omitted_redundant_stems；不会因此假定封堵仍成立，需重新做通路与完整路线检查。'
        '另外对新增树干与原生枝干做带采样余量的三维包络检查，必要时缩短或横移新增枯立木，原生枝干不变；导出网格实际使用记录的高度。'
        '不通过删改原生枝干来提高可飞性。\n\n'
        '每行参考路线交替前进，在端部空地连接换行；邻行绕障任务要求绕过封堵后返回原通道。'
        f'全局参考使用地图真值作为教师规划依据，保留 {clearance_text} 中心净空和连续轨迹检查。'
        '这些通道边界和任务门仅用于生成审核，不得作为学生 BEV 的真值通道遮罩。\n\n'
        f'当前综合状态：`{summary}`。完整参考路线通过数：`{references}`。'
        '预设障碍类型检查与完整路线安全检查分别记录；已有安全参考仍可能因预设双侧选择实际只通一侧而为 REVIEW。'
        '正式目录只发布 PASS 地图；候选失败的完整参数与原因集中保留于 candidate_reviews.jsonl，没有更换森林种子或隐藏失败。'
        '仅调整新增障碍的布局候选，训练/验证/测试的原生森林划分不变。\n\n'
        '文件只有一层地图目录，如 `train/000/`：`overview.png` 为地形俯视与完整参考路线；'
        '`scene.usd`、`scene_mesh.npz`、`tex.jpg` 是同一场景的显示/检查资源；`terrain.npz` 为教师地形；'
        '`map.json` 包含全部参数及检查；有完整安全参考的地图另有 `reference_trajectory.npz`。'
        '根目录 manifest.json 固定划分，generation.log 保留生成过程。\n\n'
        '下一阶段才采集 LiDAR、生成三层 BEV 与固定 6 m 的六点监督。在线任务管理、LiDAR 地形目标估计、观测覆盖与制动检查、实际闭环尚待验证。'
        '不得将本批地图标为已采集或直接训练。\n\n'
        '新版 map.json 的 reference_phases 将整条参考轨迹分为 INSPECT、EXIT、TURN、ENTRY；时间从参考起飞稳定后开始，'
        '采集时须加上 settle_seconds。各阶段连续覆盖完整参考时长。换行原始扫描应保留，训练样本需按阶段及任务条件另行审核；'
        'default_navigation_training 只是筛选建议，不是可训练认证。全部边界和阶段路线属于离线教师元数据。\n')

def add_arguments(p):
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES),help='Cycle selected compositions across independent forests (not a seed/profile cross product)')
    p.add_argument('--counts',nargs=3,type=int,metavar=('TRAIN','VAL','TEST'),default=None)
    p.add_argument('--seed-start',type=int,default=None)
    p.add_argument('--seeds',nargs='+',type=int,help='Explicit independent seeds for one split')
    p.add_argument('--split',choices=['train','val','test'],default=None)
    p.add_argument('--dry-run',action='store_true',help='Print map assignments without writing files or generating geometry')
    p.add_argument('--layout',choices=['variable','legacy'],default='variable')
    p.add_argument('--row-length',nargs=2,type=float,default=[65.,75.],metavar=('MIN','MAX'))
    p.add_argument('--row-start-jitter',type=float,default=2.,help='Half-range of row-start variation in meters')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/generated_multirow')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--max-attempts',type=int,default=3)
    p.add_argument('--only',nargs='+',help='Generate only selected map IDs from this batch')
    p.add_argument('--retry-failed',action='store_true',help='Continue candidate numbering for unpublished failures; preserve all published maps')
    p.add_argument('--clearance-profile',choices=PROFILE_NAMES,default='sim',
                   help='Generation clearance: sim (1.0 m, from the navigation envelope) or legacy 0.75 m sphere')
    p.add_argument('--like',type=Path,help='Reuse path/split/seed/profile of every map under this maps directory')
    p.add_argument('--complete-reference-audit',action='store_true',
                   help=argparse.SUPPRESS)
def main(argv=None,args=None):
    if args is None:
        p=argparse.ArgumentParser(description=__doc__)
        add_arguments(p)
        args=p.parse_args(argv)
    a=args
    layout=layout_options(a.layout,a.row_length,a.row_start_jitter)
    if a.like is not None and (a.seeds is not None or a.counts is not None or a.seed_start is not None):
        raise ValueError('--like cannot be combined with --seeds, --counts or --seed-start')
    if a.seeds is not None and (a.counts is not None or a.seed_start is not None):
        raise ValueError('--seeds cannot be combined with --counts or --seed-start')
    if a.split is not None and a.seeds is None:raise ValueError('--split requires --seeds')
    if a.complete_reference_audit:raise ValueError('Historical audit mode is no longer supported; use audit_forest_multirow_dataset.py')
    if a.max_attempts<1 or a.workers<1:raise ValueError('workers and max-attempts must be positive')
    entries=(like_records(a.like.resolve()) if a.like is not None else
             records(counts=a.counts if a.counts is not None else (20,5,5),
                     seed_start=a.seed_start if a.seed_start is not None else 20261700,
                     profiles=a.profiles,seeds=a.seeds,split=a.split or 'train'))
    clearance=clearance_profile(a.clearance_profile)
    if a.only and not set(a.only)<={e['path'] for e in entries}:raise ValueError('Unknown map id')
    if a.dry_run:
        print(json.dumps([e for e in entries if not a.only or e['path'] in a.only],ensure_ascii=False,indent=2));return
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    sources=['scripts/build_forest_multirow_native.py','forest_generator/multirow_tasks.py',
             'forest_generator/generator.py','forest_generator/config.json',
             'forest_generator/multirow_layout.py','forest_generator/multirow_routes.py',
             'forest_generator/terrain.py','omni_drones/navigation/forest/trajectory.py',
             'omni_drones/navigation/forest/esdf.py','forest_generator/clearance_profiles.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['maps']!=entries:
            raise ValueError('Frozen split changed')
        if manifest.get('layout',layout_options('legacy'))!=layout:
            raise ValueError('Frozen layout changed; use a new output directory')
        if manifest.get('clearance_profile',clearance_profile('legacy'))!=clearance:
            raise ValueError('Frozen clearance profile changed; use a new output directory')
        if manifest['source_hashes']!=hashes and not a.complete_reference_audit:
            raise ValueError('Frozen split or generation source changed; explicitly audit before resuming')
        if a.complete_reference_audit:
            for source in ['forest_generator/generator.py','forest_generator/config.json','forest_generator/multirow_tasks.py']:
                if manifest['source_hashes'][source]!=hashes[source]:raise ValueError('Geometry source changed')
            manifest['reference_completion_audit']=dict(source_hashes=hashes,
                changes='Continue reference checking after task semantic failures; assert saved mesh exactly identical; no relaxation of safety or gates')
    else:manifest=dict(version=6,maps=entries,layout=layout,clearance_profile=clearance,source_hashes=hashes,results=[],model_evaluation_run=False,
                       revision='variable native row extents, continuous headlands, explicit reference phases; unchanged biological distributions')
    # Import pilot results, checking their identity instead of overwriting them.
    done={r['path'] for r in manifest['results']}
    for e in entries:
        path=root/e['path']/'map.json'
        if e['path'] in done or not path.exists():continue
        info=json.loads(path.read_text())
        assert (info['parameters']['seed'],info['split'],info['tasks']['profile'])==(e['seed'],e['split'],e['profile'])
        assert info['validation'].get('reason')!='reference_not_yet_checked'
        manifest['results'].append(dict(e,status='PASS' if info['validation']['passed'] else 'REVIEW',
            validation=info['validation'],geometry_id=info['geometry_id'],
            original_trees=info['provenance']['source_tree_count'],added_trees=info['tasks']['inserted_trees'],
            aisles=len(info['lane_centers_y']),task_assignments=info['tasks']['assignments'],events=info['tasks']['events']))
        done.add(e['path'])
    render(root,manifest)
    eligible=({r['path'] for r in manifest['results'] if not r['validation']['passed']}
              if a.complete_reference_audit else {e['path'] for e in entries if e['path'] not in done})
    if a.retry_failed:eligible|={r['path'] for r in manifest['results'] if not r['validation']['passed']}
    if a.only:
        if not set(a.only)<={e['path'] for e in entries}:raise ValueError('Unknown map id')
        eligible &= set(a.only)
    completed=0
    prior={r['path']:r for r in manifest['results']}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={}
        for entry in entries:
            if entry['path'] not in eligible:continue
            previous=prior.get(entry['path'],{})
            last=max((c['attempt'] for c in previous.get('candidate_reviews',[])),default=-1)
            start=previous.get('last_attempt',last)+1
            future=pool.submit(worker,str(root),entry,a.complete_reference_audit,a.max_attempts,start,layout,a.clearance_profile)
            futures[future]=entry
        for future in as_completed(futures):
            result,log,reviews=future.result()
            if reviews:
                with (root/'candidate_reviews.jsonl').open('a') as f:
                    for review in reviews:f.write(json.dumps(review,ensure_ascii=False,allow_nan=False)+'\n')
            result['candidate_reviews']=prior.get(result['path'],{}).get('candidate_reviews',[])+result.get('candidate_reviews',[])
            manifest['results']=[r for r in manifest['results'] if r['path']!=result['path']]+[result]
            manifest['results'].sort(key=lambda r:entries.index(next(e for e in entries if e['path']==r['path'])))
            with (root/'generation.log').open('a') as f:f.write(f'\n{result["path"]} seed={result["seed"]}\n'+log)
            render(root,manifest)
            completed+=1
            print(f'{completed}/{len(eligible)} {result["path"]}: {result["status"]} {result["validation"].get("reason","")}',flush=True)
    ids=[r['geometry_id'] for r in manifest['results'] if 'geometry_id' in r]
    assert len(ids)==len(set(ids)), 'Duplicate geometry across maps'
    selected=set(a.only) if a.only else {e['path'] for e in entries}
    if any(r['path'] in selected and r['status']!='PASS' for r in manifest['results']):
        raise SystemExit(1)

if __name__=='__main__':main()
