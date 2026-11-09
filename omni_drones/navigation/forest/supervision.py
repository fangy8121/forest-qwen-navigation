"""Coordinate-aware answer supervision for the six-point planning protocol.

Weights follow the exact JSON character spans and the fast tokenizer's offsets.
They never infer coordinate tokens from their decoded text.  A token covering
both a coordinate and JSON punctuation receives the character-weighted average;
punctuation, status, and the assistant end marker keep their normal weight.
"""
import json
import math
import re


MANEUVER_THRESHOLD = .1  # Half of the observation's 0.2 m grid cell.
_STRING_OR_NUMBER = re.compile(
    r'"(?:\\.|[^"\\])*"|(?P<number>-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)',
    re.DOTALL,
)
_PROMPT_GOAL = re.compile(r'目标XYZ\s*=\s*(\[[^\]]*\])')


def _finite_xyz(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)):
        raise ValueError(f'{name} must contain three finite numbers')
    try:
        xyz = [float(x) for x in value]
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must contain three finite numbers') from exc
    if not all(math.isfinite(x) for x in xyz):
        raise ValueError(f'{name} must contain three finite numbers')
    return xyz


def parse_prompt_goal(prompt):
    """Read the one explicit ``目标XYZ=[x,y,z]`` field; refuse unknown formats."""
    matches = list(_PROMPT_GOAL.finditer(prompt))
    if len(matches) != 1:
        raise ValueError('prompt must contain exactly one explicit 目标XYZ=[x,y,z] field')
    try:
        value = json.loads(matches[0].group(1))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError('invalid 目标XYZ field in prompt') from exc
    return _finite_xyz(value, 'prompt goal')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate answer JSON key')
        result[key] = value
    return result


def _coordinate_spans(answer, goal, coordinate_weight, maneuver_multiplier):
    """Return (start, end, weight) for the 18 GO coordinate numbers."""
    try:
        parsed = json.loads(answer, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError('answer must be a valid six-point protocol JSON object') from exc
    if not isinstance(parsed, dict) or set(parsed) != {'status', 'points'}:
        raise ValueError('answer must contain exactly status and points')
    if parsed['status'] == 'HOLD':
        if parsed['points'] != []:
            raise ValueError('HOLD answer must have no points')
        return []
    if parsed['status'] != 'GO' or not isinstance(parsed['points'], list) or len(parsed['points']) != 6:
        raise ValueError('GO answer must contain exactly six XYZ points')
    points = [_finite_xyz(point, 'GO point') for point in parsed['points']]
    # Supplying the task goal also detects a turn concentrated in the final point.
    # Endpoint fallback only measures curvature relative to the output chord.
    reference = _finite_xyz(goal, 'goal') if goal is not None else points[-1]
    norm2 = sum(x*x for x in reference)
    numbers = [m for m in _STRING_OR_NUMBER.finditer(answer) if m.group('number') is not None]
    if len(numbers) != 18:
        raise ValueError('cannot align the 18 GO coordinate number spans')
    spans = []
    for i, point in enumerate(points):
        fraction = sum(x*y for x, y in zip(point, reference))/norm2 if norm2 > 1e-12 else 0.
        for axis, value in enumerate(point):
            weight = coordinate_weight
            if axis in (1, 2) and abs(value-fraction*reference[axis]) >= MANEUVER_THRESHOLD:
                weight *= maneuver_multiplier
            match = numbers[3*i+axis]
            spans.append((match.start(), match.end(), weight))
    return spans


def answer_token_weights(tokenizer, input_ids, labels, answer, coordinate_weight=4.,
                         maneuver_multiplier=2., goal=None):
    """Build ``[1,T]`` weights compatible with ``qwen.encode_sample`` labels.

    Masked prompt/image/padding positions receive zero.  The answer and its end
    tokens receive one, with GO coordinate spans receiving stronger supervision.
    A supervised token must exactly match the corresponding answer token ID;
    changed templates or boundaries fail instead of silently shifting weights.
    ``goal`` should be the prompt's local XYZ goal.  Without it, key y/z values
    are measured against the output's last point, which can miss final-point turns.
    """
    for name, value in [('coordinate_weight', coordinate_weight),
                        ('maneuver_multiplier', maneuver_multiplier)]:
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    spans = _coordinate_spans(answer, goal, coordinate_weight, maneuver_multiplier)
    return _span_token_weights(tokenizer, input_ids, labels, answer, spans)


def spatial_token_weights(tokenizer, input_ids, labels, answer, state_weight=1.):
    """Weight the actual category choice, preserving JSON/end-token supervision."""
    from .grounding import parse_spatial
    if isinstance(state_weight, bool) or not math.isfinite(state_weight) or state_weight <= 0:
        raise ValueError('state_weight must be finite and positive')
    parsed = parse_spatial(answer)
    strings = list(_STRING_OR_NUMBER.finditer(answer))
    if (len(strings) != 2 or json.loads(strings[0].group()) != 'state'
            or json.loads(strings[1].group()) != parsed['state']):
        raise ValueError('cannot align the spatial state value')
    value = strings[1]
    return _span_token_weights(tokenizer, input_ids, labels, answer,
                               [(value.start()+1, value.end()-1, state_weight)])


def balanced_path_token_weights(tokenizer, input_ids, labels, answer,
                                decision_fraction=.25, coordinate_weight=4., goal=None):
    """Give the first GO/HOLD decision an equal loss share for either status.

    Normalizing long coordinate answers and short HOLD answers separately can
    otherwise give HOLD decisions much more influence per example. Allocate a
    fixed fraction to the first status token; preserve relative coordinate,
    syntax and end-token weights in the remaining fraction. Split spellings
    (e.g. H + OLD) still have exactly one category decision.
    """
    import torch
    if isinstance(decision_fraction,bool) or not math.isfinite(decision_fraction) or not 0<decision_fraction<1:
        raise ValueError('decision_fraction must be strictly between zero and one')
    weights=answer_token_weights(tokenizer,input_ids,labels,answer,coordinate_weight,1.,goal)
    match=re.search(r'"status"\s*:\s*"(GO|HOLD)"',answer)
    if match is None:raise ValueError('missing path status value')
    aligned=_span_token_weights(tokenizer,input_ids,labels,answer,[(match.start(1),match.end(1),2.)])
    positions=torch.nonzero(aligned[0]>1.,as_tuple=False).flatten()
    if not len(positions):raise ValueError('missing path decision token')
    position=positions[0];weights[0,position]=0.
    denominator=weights.sum()
    if denominator<=0:raise ValueError('missing path continuation tokens')
    weights*=((1.-decision_fraction)/denominator)
    weights[0,position]=decision_fraction
    return weights


def spatial_decision_token_weights(tokenizer, input_ids, labels, answer, state_weight=1.):
    """Give each category one equally weighted decision token.

    Qwen tokenizes ``observed`` as two tokens, while ``blocked`` and ``unknown``
    each use one. Weighting all value tokens and normalizing per answer otherwise
    gives the first observed-choice token less influence. Continuation spelling,
    JSON syntax and end markers retain their ordinary weight.
    """
    import torch
    if isinstance(state_weight, bool) or not math.isfinite(state_weight) or state_weight <= 0:
        raise ValueError('state_weight must be finite and positive')
    aligned = spatial_token_weights(tokenizer, input_ids, labels, answer, 2.)
    decisions = torch.nonzero(aligned[0] > 1., as_tuple=False).flatten()
    if not len(decisions):
        raise ValueError('missing spatial category decision token')
    weights = (labels != -100).float()
    weights[0, decisions[0]] = state_weight
    return weights


def _span_token_weights(tokenizer, input_ids, labels, answer, spans):
    """Align semantic character spans with the exact supervised token boundary."""
    import torch
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('answer supervision requires a fast tokenizer with offsets')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or labels.shape != input_ids.shape:
        raise ValueError('answer weights require a single encoded sample with [1,T] IDs and labels')
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if not ids or len(ids) != len(offsets):
        raise ValueError('empty answer or invalid tokenizer offsets')
    active = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    if len(active) < len(ids):
        raise ValueError('answer extends beyond supervised labels')
    start = int(active[0])
    positions = torch.arange(start, start+len(ids), device=labels.device)
    expected = torch.tensor(ids, dtype=input_ids.dtype, device=input_ids.device)
    if (not torch.equal(active[:len(ids)], positions) or
            not torch.equal(input_ids[0, start:start+len(ids)], expected) or
            not torch.equal(labels[0, start:start+len(ids)].to(expected.device), expected)):
        raise ValueError('answer token IDs do not match the encoded assistant-label boundary')
    weights = (labels != -100).to(torch.float32)
    token_values = []
    for begin, end in offsets:
        if not 0 <= begin < end <= len(answer):
            raise ValueError('answer tokenizer returned invalid character offsets')
        excess = sum(max(0, min(end, right)-max(begin, left))*(weight-1.)
                     for left, right, weight in spans)
        token_values.append(1.+excess/(end-begin))
    weights[0, start:start+len(ids)] = torch.tensor(token_values, dtype=torch.float32, device=weights.device)
    return weights


def weighted_causal_cross_entropy(logits, labels, token_weights, chunk_size=32):
    """Mean causal CE over supervised tokens, normalized by their total weight.

    Labels and weights belong to the *target* position: logits at t predict label
    and weight at t+1.  Only supervised logits are gathered, once, in their native
    dtype, then cast in small chunks.  This avoids full-sequence float32 copies
    and repeated full-sequence gather gradients during backward.
    All-one supervised weights reproduce ordinary mean causal cross entropy.
    """
    import torch
    import torch.nn.functional as F
    if (logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape or
            token_weights.shape != labels.shape or labels.shape[1] < 2):
        raise ValueError('expected logits [B,T,V] and labels/weights [B,T], with T >= 2')
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0):
        raise ValueError('chunk_size must be a positive integer')
    if not torch.isfinite(token_weights).all() or (token_weights < 0).any():
        raise ValueError('token weights must be finite and nonnegative')
    targets, weights = labels[:, 1:], token_weights[:, 1:]
    positions = torch.nonzero((targets != -100) & (weights > 0), as_tuple=False)
    if not len(positions):
        raise ValueError('no supervised next-token targets with positive weight')
    batch_indices, time_indices = positions.unbind(1)
    selected_weights = weights[batch_indices, time_indices].float()
    selected_logits = logits[batch_indices, time_indices, :]
    selected_targets = targets[batch_indices, time_indices]
    denominator = selected_weights.sum()
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(positions), chunk_size):
        losses = F.cross_entropy(selected_logits[start:start+chunk_size].float(),
                                 selected_targets[start:start+chunk_size], reduction='none')
        total = total+(losses*selected_weights[start:start+chunk_size]).sum()
    return total/denominator
