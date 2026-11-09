"""Ask a Qwen worker every item x candidate order of the decision test set.

Reports accuracy against the teacher (non-trivial items), per category,
consistency across candidate orders, letter distribution and latency.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omni_drones.navigation.forest.coverage_prompt import point_choice_prompt  # noqa: E402
from omni_drones.navigation.forest.qwen_subgoal import SubgoalModelProcess  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', default=str(ROOT/'outputs/03_Qwen点选决策/测试集'))
    parser.add_argument('--model', default=str(ROOT/'models/Qwen3.5-4B'))
    parser.add_argument('--facts', action='store_true', help='append observed aisle facts to candidates')
    parser.add_argument('--name', default=None)
    parser.add_argument('--include-trivial', action='store_true')
    parser.add_argument('--images', choices=('both', 'global', 'bev', 'none'), default='both',
                        help='none sends one blank image (the worker requires one)')
    parser.add_argument('--text', choices=('full', 'no_state'), default='full',
                        help='no_state drops where the vehicle is and the aisle table: position must come from images')
    parser.add_argument('--image-max-pixels', type=int, default=262144)
    parser.add_argument('--letter-scores', action='store_true',
                        help='also score each item by its candidates\' letter probabilities summed over the orders')
    args = parser.parse_args()
    bench = Path(args.bench)
    name = args.name or '_'.join([Path(args.model).name, 'facts' if args.facts else 'base']
                                 + ([args.images] if args.images != 'both' else [])
                                 + ([args.text] if args.text != 'full' else []))
    out = bench/'results'/name
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((bench/'index.json').read_text())['items']
    items = [json.loads(Path(row['path']).read_text()) for row in index if args.include_trivial or not row['trivial']]
    worker = SubgoalModelProcess(str(ROOT/'.venvs/forest_qwen35/bin/python'), args.model, out/'worker',
        device='cuda:0', max_input_tokens=8192, max_new_tokens=48, image_max_pixels=args.image_max_pixels,
        load_timeout=300., inference_timeout=60.)
    blank = out/'blank.png'
    from PIL import Image
    Image.new('RGB', (256, 256), (150, 156, 162)).save(blank)
    rows = []
    try:
        for item in items:
            payload = json.loads((Path(item['source'])/'input.json').read_text())
            state = payload['state']
            public = {p['candidate_id']: p for p in state.get('candidates', [])}
            by_id = {c['candidate_id']: c for c in payload['candidates']}
            for k, variant in enumerate(item['variants']):
                ordered = [by_id[cid] for cid in variant['order']]
                prompt = point_choice_prompt(state, ordered, variant['labels'], public, facts=args.facts)
                if args.text == 'no_state':
                    prompt = '\n'.join(line for line in prompt.split('\n')
                                       if not line.startswith(('现在：', '上一段：', '通道：', '要求的通道已全部巡完')))
                images = dict(both=variant['images'], global_=variant['images'][:1], bev=variant['images'][1:],
                              none=[str(blank)])[args.images if args.images != 'global' else 'global_']
                assert worker.submit(images, prompt, request_id=f'{item["id"]}_p{k}',
                                     response_schema='choice', letter_scores=args.letter_scores)
                while (reply := worker.poll()) is None:
                    time.sleep(.02)
                raw = reply.get('raw') or ''
                match = re.search(r'"choice"\s*:\s*"([A-Z])"', raw)
                letter = match.group(1) if match else None
                pick = {v: c for c, v in variant['labels'].items()}.get(letter)
                letters = reply.get('letter_probabilities')
                rows.append(dict(item=item['id'], category=item['category'], permutation=k, letter=letter,
                                 pick=pick, correct=pick in item['acceptable'], valid=reply.get('error') is None,
                                 raw=raw.strip(), latency=reply.get('latency_seconds'),
                                 tokens=reply.get('input_tokens'), options=len(variant['labels']),
                                 correct_letters=sorted(variant['labels'][c] for c in item['acceptable']
                                                        if c in variant['labels']),
                                 scores=None if letters is None else
                                 {c: letters[v] for c, v in variant['labels'].items()}))
                print(f'{item["id"]:34s} p{k} {letter} {"OK " if rows[-1]["correct"] else "BAD"} {pick}', flush=True)
    finally:
        worker.close()
    by_item = defaultdict(list)
    for row in rows:
        by_item[row['item']].append(row)
    categories = defaultdict(lambda: [0, 0])
    for row in rows:
        categories[row['category']][0] += row['correct']
        categories[row['category']][1] += 1
    first = Counter(row['letter'] for row in rows)
    summary = dict(model=args.model, facts=args.facts, queries=len(rows),
        accuracy=round(sum(r['correct'] for r in rows)/max(1, len(rows)), 3),
        item_all_orders_correct=round(sum(all(r['correct'] for r in v) for v in by_item.values())/max(1, len(by_item)), 3),
        same_pick_across_orders=round(sum(len({r['pick'] for r in v}) == 1 for v in by_item.values())/max(1, len(by_item)), 3),
        valid_format=round(sum(r['valid'] for r in rows)/max(1, len(rows)), 3),
        by_category={k: f'{a}/{b}' for k, (a, b) in sorted(categories.items())},
        letters=dict(sorted(first.items(), key=lambda x: str(x[0]))),
        latency_mean=round(sum(r['latency'] or 0 for r in rows)/max(1, len(rows)), 3),
        tokens_mean=round(sum(r['tokens'] or 0 for r in rows)/max(1, len(rows)), 1),
        wrong_letters=dict(sorted(Counter(r['letter'] for r in rows if not r['correct']).items(),
                                  key=lambda x: str(x[0]))),
        accuracy_by_first_correct_letter={letter: f'{sum(r["correct"] for r in group)}/{len(group)}'
            for letter, group in sorted(_group(rows, lambda r: (r['correct_letters'] or ['-'])[0]).items())})
    if args.letter_scores:
        summary['averaged'] = averaged(by_item, {item['id']: item for item in items})
    (out/'rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


def _group(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def averaged(by_item, items):
    """Item accuracy when one decision sums each candidate's letter probability over orders.

    Each order's probabilities are normalised over its offered letters, so a
    letter-position bias adds the same to every candidate across rotations.
    Compared with the first order alone (the live order) and with each order
    alone; ``orders`` limits how many rotations are combined.
    """
    def pick(rows):
        total = defaultdict(float)
        for row in rows:
            mass = sum(row['scores'].values()) or 1.
            for candidate, p in row['scores'].items():
                total[candidate] += p/mass
        return max(total, key=total.get) if total else None

    result = {}
    scored = {i: sorted(v, key=lambda r: r['permutation']) for i, v in by_item.items()
              if all(r['scores'] for r in v)}
    for name, rule in (('first_order', lambda v: v[:1]), ('two_orders', lambda v: v[:2]), ('all_orders', lambda v: v)):
        categories = defaultdict(lambda: [0, 0])
        for item_id, rows in scored.items():
            correct = pick(rule(rows)) in items[item_id]['acceptable']
            categories[rows[0]['category']][0] += correct
            categories[rows[0]['category']][1] += 1
        hits, count = sum(a for a, _ in categories.values()), sum(b for _, b in categories.values())
        result[name] = dict(accuracy=round(hits/max(1, count), 3), items=count,
                            by_category={k: f'{a}/{b}' for k, (a, b) in sorted(categories.items())})
    single = [r['correct'] for v in scored.values() for r in v]
    result['single_order_mean'] = round(sum(single)/max(1, len(single)), 3)
    result['unscored_items'] = len(by_item)-len(scored)
    return result


if __name__ == '__main__':
    main()
