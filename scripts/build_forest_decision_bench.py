"""Build the point_choice decision test set from saved runs and synthetic scenes.

Each item keeps the saved decision state, the teacher's acceptable candidates
and ``--permutations`` relabelled image pairs. Prompts are rebuilt at
evaluation time, so prompt variants need no rebuild.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omni_drones.navigation.forest.decision_bench import build_item, synthetic_decision  # noqa: E402

S_COMMAND = '沿S形逐条巡检整个树林，在已确认的行末进入相邻未巡通道反向巡检，补齐遗漏，完成后返回起点并停稳。'


def s_state(k):
    """After S coverage of aisles 1..k: the end reached and the last direction."""
    direction = 1 if k % 2 else -1
    return dict(covered=tuple(range(1, k+1)), current=k, end='high' if direction > 0 else 'low', previous=direction)


def scenes():
    """(id, category, kwargs) for the synthetic part of the set."""
    out = []
    for aisles in (5, 6, 7):
        for yaw in (0., np.pi/2, -np.pi/2, np.pi):
            out.append((f'start_a{aisles}_y{int(np.rad2deg(yaw))}', 'start', dict(aisles=aisles, yaw=yaw, seed=aisles)))
    for aisles in (5, 7):
        for k in range(1, aisles):
            out.append((f'change_a{aisles}_k{k}', 'change', dict(aisles=aisles, seed=k, **s_state(k))))
    # S started from the far side: aisles N..k covered, next is k-1.
    for k in (5, 4, 3, 2):
        covered = tuple(range(k, 7))
        direction = 1 if (6-k) % 2 == 0 else -1
        out.append((f'change_reverse_k{k}', 'change', dict(aisles=6, covered=covered, current=k,
                    end='high' if direction > 0 else 'low', previous=direction, seed=10+k)))
    # An aisle left behind: both nearest open aisles are acceptable.
    out.append(('change_gap_4', 'change', dict(aisles=6, covered=(1, 2, 4), current=4, end='high', previous=1, seed=21)))
    out.append(('change_gap_2', 'change', dict(aisles=6, covered=(1, 2, 4, 5), current=2, end='low', previous=-1, seed=22)))
    for k, (a, b), x in ((2, (24., 50.), 50.), (3, (60., 90.), 60.), (4, (24., 40.), 40.), (1, (70., 90.), 70.)):
        direction = 1 if a <= 24. else -1
        out.append((f'inspect_k{k}', 'inspect', dict(aisles=6, partial=(k, a, b), current=k,
                    position=[x, 3.+5.*k-2.5, 2.5], yaw=0. if direction > 0 else np.pi, seed=30+k,
                    covered=tuple(range(1, k)))))
    for subset, k in (([1, 3, 5], 1), ([1, 3, 5], 3), ([2, 4, 6], 2), ([1, 4, 6], 4), ([2, 5], 2)):
        covered = tuple(i for i in subset if i <= k)
        direction = 1 if len(covered) % 2 else -1
        out.append((f'subset_{"".join(map(str, subset))}_k{k}', 'subset', dict(aisles=6, covered=covered, current=k,
                    end='high' if direction > 0 else 'low', previous=direction, subset=subset, order='free',
                    command=f'只巡检第{"、".join(map(str, subset))}条通道，完成后返回起点并停稳。', seed=40+k)))
    for k in (2, 4):
        out.append((f'free_k{k}', 'free', dict(aisles=6, order='free', seed=50+k, **s_state(k))))
    for current, target in ((1, 4), (3, 6), (2, 5), (4, 1)):
        out.append((f'visit_{current}_to_{target}', 'visit', dict(aisles=6, intent='VISIT_CORRIDOR', visit=target,
                    order='S', command=f'去第{target}条通道巡检。', seed=60+target, **s_state(current))))
    for aisles in (5, 6):
        out.append((f'return_a{aisles}', 'return', dict(aisles=aisles, seed=70+aisles, **s_state(aisles))))
    return out


def run_context(run):
    config = json.loads((run/'config.json').read_text())
    terrain = None
    source = json.loads((run/'map_source.json').read_text())
    if config['coverage'].get('terrain_source') == 'truth':
        t = np.load(Path(source['path'])/'terrain.npz')
        terrain = {k: t[k].tolist() for k in ('xs', 'ys', 'heights')}
    return dict(flight=config['task']['flight'], operating_bounds=config['coverage']['operating_bounds'],
                height_bounds=config['coverage']['height_bounds'],
                planning_clearance_reserve=config['lower'].get('planning_clearance_reserve', 0.), terrain=terrain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='*', default=None, help='run folders (default: all point_choice runs)')
    parser.add_argument('--out', default=str(ROOT/'outputs/03_Qwen点选决策/测试集'))
    parser.add_argument('--permutations', type=int, default=3)
    parser.add_argument('--global-pixels', type=int, default=None, help='overview pixel budget (default 262144)')
    parser.add_argument('--global-width', type=int, default=None)
    parser.add_argument('--crop', choices=('work_area', 'structure'), default='work_area')
    args = parser.parse_args()
    view = dict(crop=args.crop, **({'max_pixels': args.global_pixels} if args.global_pixels else {}),
                **({'max_width': args.global_width} if args.global_width else {}))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out/'items', ignore_errors=True)
        shutil.rmtree(out/'synthetic', ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in args.runs] if args.runs is not None else sorted(
        p.parents[2] for p in (ROOT/'outputs/01_森林巡检仿真跑通/runs').glob('*/language_decisions/decision_000/structure.json'))
    index, skipped = [], {}
    for run in runs:
        context = run_context(run)
        for folder in sorted((run/'language_decisions').glob('decision_*')):
            if not (folder/'replay.npz').exists():
                continue
            item_id = f'real_{run.name[-8:]}_{folder.name[-3:]}'
            item, reason = build_item(folder, item_id, out/'items'/item_id, context, args.permutations,
                                      global_view=view)
            if item is None:
                skipped[item_id] = reason
            else:
                index.append(dict(id=item_id, path=str(out/'items'/item_id/'item.json'), category=item['category'],
                                  trivial=item['trivial'], source='real'))
    for item_id, category, kwargs in scenes():
        folder, context = synthetic_decision(out/'synthetic'/item_id, **kwargs)
        item, reason = build_item(folder, 'syn_'+item_id, out/'items'/('syn_'+item_id), context, args.permutations,
                                  category_hint=category, global_view=view)
        if item is None:
            skipped['syn_'+item_id] = reason
        else:
            index.append(dict(id=item['id'], path=str(out/'items'/item['id']/'item.json'), category=category,
                              trivial=item['trivial'], source='synthetic'))
    (out/'index.json').write_text(json.dumps(dict(items=index, skipped=skipped), ensure_ascii=False, indent=1))
    counts = {}
    for row in index:
        counts[row['category']] = counts.get(row['category'], 0)+1
    print(json.dumps(dict(items=len(index), nontrivial=sum(not r['trivial'] for r in index), by_category=counts,
                          skipped=len(skipped)), ensure_ascii=False))


if __name__ == '__main__':
    main()
