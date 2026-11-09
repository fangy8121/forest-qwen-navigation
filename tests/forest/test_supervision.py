"""Alignment and gradients for coordinate-aware six-point answer supervision."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from omni_drones.navigation.forest.supervision import (
    answer_token_weights, parse_prompt_goal, spatial_token_weights,
    spatial_decision_token_weights, weighted_causal_cross_entropy,
    balanced_path_token_weights,
)


class OffsetTokenizer:
    is_fast = True

    def __init__(self, parts):
        self.parts = parts

    def __call__(self, text, **kwargs):
        assert text == ''.join(self.parts)
        assert kwargs == dict(add_special_tokens=False, return_offsets_mapping=True)
        offsets, cursor = [], 0
        for part in self.parts:
            offsets.append((cursor, cursor+len(part)))
            cursor += len(part)
        return dict(input_ids=list(range(10, 10+len(self.parts))), offset_mapping=offsets)


def encoded(tokenizer, answer):
    tokens = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)['input_ids']
    # Prompt/image tokens, assistant answer, im_end/newline, and right padding.
    ids = torch.tensor([[1, 2]+tokens+[3, 4, 0]])
    labels = ids.clone()
    labels[0, :2] = -100
    labels[0, -1] = -100
    return ids, labels


def go(points=None):
    points = points if points is not None else [[i, 0, 0] for i in range(1, 7)]
    return json.dumps(dict(status='GO', points=points), separators=(',', ':'))


def test_prompt_goal_requires_one_finite_xyz_field():
    assert parse_prompt_goal('前文目标XYZ=[6, -0.25, 1e-1]；后文') == [6., -.25, .1]
    for prompt in ['goal=[1,2,3]', '目标XYZ=[1,2]', '目标XYZ=[1,2,true]',
                   '目标XYZ=[1,2,NaN]', '目标XYZ=[1,2,3] 目标XYZ=[4,5,6]',
                   '目标XYZ=[1,2,inf]']:
        with pytest.raises(ValueError):
            parse_prompt_goal(prompt)


def test_goal_marks_final_turn_and_leaves_status_prompt_end_padding_normal():
    answer = go([[i, 0, 0] for i in range(1, 6)]+[[6, 2, .2]])
    tokenizer = OffsetTokenizer(list(answer))
    ids, labels = encoded(tokenizer, answer)
    weights = answer_token_weights(tokenizer, ids, labels, answer, goal=[6, 0, 0])
    assert weights.dtype == torch.float32
    assert torch.equal(weights[0, :2], torch.zeros(2))
    assert weights[0, -1] == 0
    assert weights[0, -3:-1].tolist() == [1., 1.]
    assert weights[0, answer.index('GO')+2] == 1
    last_turn = answer.rindex('[6,2,0.2]')
    assert weights[0, last_turn+2+1] == 4  # x coordinate.
    assert weights[0, last_turn+2+3] == 8  # final y deviation.
    assert weights[0, last_turn+2+5] == 8  # final z deviation.
    assert weights[0, last_turn+2+2] == 1  # comma.
    fallback = answer_token_weights(tokenizer, ids, labels, answer)
    assert fallback[0, last_turn+2+3] == 4  # Endpoint chord misses this final turn.


def test_tokens_mixing_two_numbers_and_punctuation_use_character_fraction():
    answer = go([[1, -2, 0]]+[[i, 0, 0] for i in range(2, 7)])
    begin = answer.index('1,-2')
    parts = [answer[:begin], '1,-2', answer[begin+4:]]
    tokenizer = OffsetTokenizer(parts)
    ids, labels = encoded(tokenizer, answer)
    weights = answer_token_weights(tokenizer, ids, labels, answer, goal=[6, 0, 0])
    # x: one character at weight 4; comma: one at 1; y: two at 8.
    assert weights[0, 3] == pytest.approx((4+1+8*2)/4)


def test_hold_status_and_end_tokens_are_unweighted():
    answer = '{"status":"HOLD","points":[]}'
    tokenizer = OffsetTokenizer(list(answer))
    ids, labels = encoded(tokenizer, answer)
    weights = answer_token_weights(tokenizer, ids, labels, answer)
    torch.testing.assert_close(weights, (labels != -100).float())


def test_path_decision_gradient_is_independent_of_answer_length():
    gradients=[]
    for answer,status in [(go(),'GO'),('{"status":"HOLD","points":[]}','HOLD')]:
        tokenizer=OffsetTokenizer(list(answer));ids,labels=encoded(tokenizer,answer)
        weights=balanced_path_token_weights(tokenizer,ids,labels,answer)
        position=2+answer.index(status)
        assert weights[labels==-100].count_nonzero()==0
        assert weights[0,-2]>0  # Assistant end continuation stays supervised.
        logits=torch.zeros((1,labels.shape[1],512),requires_grad=True)
        weighted_causal_cross_entropy(logits,labels,weights).backward()
        gradients.append(logits.grad[0,position-1].norm())
    torch.testing.assert_close(gradients[0],gradients[1])


@pytest.mark.parametrize('fraction',[0,1,-.1,float('nan'),True])
def test_path_decision_fraction_rejects_invalid_values(fraction):
    answer=go();tokenizer=OffsetTokenizer(list(answer));ids,labels=encoded(tokenizer,answer)
    with pytest.raises(ValueError):
        balanced_path_token_weights(tokenizer,ids,labels,answer,decision_fraction=fraction)


@pytest.mark.parametrize('state', ['blocked', 'observed', 'unknown'])
def test_spatial_weights_follow_only_the_exact_category_value(state):
    answer = json.dumps({'state': state}, separators=(',', ':'))
    tokenizer = OffsetTokenizer(list(answer))
    ids, labels = encoded(tokenizer, answer)
    weights = spatial_token_weights(tokenizer, ids, labels, answer, 8.)
    start = answer.index(state)
    assert (weights[0, start+2:start+2+len(state)] == 8).all()
    keep = weights.clone()
    keep[0, start+2:start+2+len(state)] = 1.
    torch.testing.assert_close(keep, (labels != -100).float())
    torch.testing.assert_close(spatial_token_weights(tokenizer, ids, labels, answer),
                               (labels != -100).float())


def test_spatial_mixed_punctuation_tokens_are_weighted_by_character_overlap():
    answer = '{"state":"blocked"}'
    tokenizer = OffsetTokenizer(['{"state":', '"blocked"', '}'])
    ids, labels = encoded(tokenizer, answer)
    weights = spatial_token_weights(tokenizer, ids, labels, answer, 8.)
    assert weights[0, 3] == pytest.approx((7*8+2)/9)
    with pytest.raises(ValueError, match='finite and positive'):
        spatial_token_weights(tokenizer, ids, labels, answer, float('nan'))


def test_spatial_decision_has_one_weighted_choice_regardless_of_category_spelling_length():
    for state,parts in [('blocked',['blocked']),('observed',['obs','erved']),('unknown',['unknown'])]:
        parts = ['{"state":"']+parts+['"}']
        answer = ''.join(parts)
        tokenizer = OffsetTokenizer(parts)
        ids,labels = encoded(tokenizer,answer)
        weights = spatial_decision_token_weights(tokenizer,ids,labels,answer,8.)
        assert (weights == 8.).sum() == 1
        assert weights[0,3] == 8.
        assert (weights[0,4:-1] == 1.).all()
        torch.testing.assert_close(spatial_decision_token_weights(tokenizer,ids,labels,answer,1.),
                                   (labels != -100).float())


def test_alignment_or_invalid_protocol_fails_instead_of_silently_weighting():
    answer = go()
    tokenizer = OffsetTokenizer(list(answer))
    ids, labels = encoded(tokenizer, answer)
    ids[0, 3] += 1
    with pytest.raises(ValueError, match='boundary'):
        answer_token_weights(tokenizer, ids, labels, answer)
    for answer in ['{"status":"HOLD","points":[[1,2,3]]}',
                   '{"status":"GO","points":[[1,2,3]]}',
                   '{"status":"HOLD","status":"HOLD","points":[]}',
                   go([[1, 2, float('nan')]]*6), go([[1, 2, True]]*6)]:
        tokenizer = OffsetTokenizer(list(answer))
        ids, labels = encoded(tokenizer, answer)
        with pytest.raises(ValueError):
            answer_token_weights(tokenizer, ids, labels, answer)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_all_one_weights_match_standard_causal_ce_and_gradients(dtype):
    torch.manual_seed(7)
    logits = torch.randn(2, 9, 17, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 17, (2, 9))
    labels[:, :3] = -100
    labels[0, -2:] = -100
    weights = (labels != -100).float()
    actual = weighted_causal_cross_entropy(logits, labels, weights, chunk_size=2)
    expected = F.cross_entropy(logits[:, :-1].float().reshape(-1, 17), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad.count_nonzero()
    assert torch.count_nonzero(actual_grad[:, -1]) == 0


def test_weights_follow_shifted_target_and_normalize_out_global_scale():
    logits = torch.tensor([[[2., -1.], [-1., 3.], [1., -2.], [3., 4.]]], requires_grad=True)
    labels = torch.tensor([[-100, -100, 0, 1]])
    weights = torch.tensor([[0., 0., 4., 8.]])
    actual = weighted_causal_cross_entropy(logits, labels, weights, chunk_size=1)
    losses = F.cross_entropy(logits[0, 1:3], labels[0, 2:], reduction='none')
    expected = (losses[0]*4+losses[1]*8)/12
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, weighted_causal_cross_entropy(logits, labels, weights*7))
    actual.backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1:3]) > 0
    assert torch.count_nonzero(logits.grad[0, 3]) == 0


def test_empty_or_invalid_loss_weights_refused():
    logits = torch.zeros(1, 3, 2)
    labels = torch.full((1, 3), -100)
    with pytest.raises(ValueError, match='no supervised'):
        weighted_causal_cross_entropy(logits, labels, torch.zeros(1, 3))
    with pytest.raises(ValueError, match='finite and nonnegative'):
        weighted_causal_cross_entropy(logits, labels, torch.tensor([[0., float('nan'), 1.]]))


def test_local_qwen_fast_tokenizer_negative_signs_and_template_boundary():
    transformers = pytest.importorskip('transformers')
    model_path = Path(__file__).resolve().parents[2]/'models/Qwen3.5-2B'
    if not model_path.is_dir():
        pytest.skip('local Qwen3.5 tokenizer not installed')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    answer = go([[i, -.25, 0.] for i in range(1, 7)])
    prefix = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    ids = torch.tensor([tokenizer(prefix+answer+'<|im_end|>\n', add_special_tokens=False)['input_ids']])
    labels = ids.clone()
    labels[:, :len(prefix_ids)] = -100
    weights = answer_token_weights(tokenizer, ids, labels, answer, goal=[6, 0, 0])
    offsets = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    mixed = [i for i, (begin, end) in enumerate(offsets) if answer[begin:end] == ',-']
    assert mixed  # Actual Qwen tokenizer merges comma and negative sign.
    for i in mixed:
        assert weights[0, len(prefix_ids)+i] == pytest.approx(4.5)
    assert torch.count_nonzero(weights[labels == -100]) == 0
    assert weights[0, -1] == 1
