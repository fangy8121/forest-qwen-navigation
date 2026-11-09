"""The physical loop must survive malformed, slow or dead model workers."""
import json
from pathlib import Path
import sys
import time

import pytest

from omni_drones.navigation.forest.qwen_subgoal import (
    SubgoalModelProcess, parse_decision, parse_task_plan, parse_forest_task_plan)


@pytest.mark.parametrize('raw', [
    '{"candidate_id":"C3","behavior":"REJOIN"}',
    '```json\n{"candidate_id":"C3","behavior":"REJOIN"}\n```',
    'Answer: {"candidate_id":"C3","behavior":"REJOIN"}',
])
def test_parse_one_complete_decision(raw):
    assert parse_decision(raw) == dict(candidate_id='C3', behavior='REJOIN')


@pytest.mark.parametrize('raw', [
    '', '[]', '{"candidate_id":"C3"}',
    '{"candidate_id":"C3","behavior":"REJOIN","xyz":[0,0,0]}',
    '{"candidate_id":"","behavior":"REJOIN"}',
    '{"candidate_id":"C3","behavior":"GO"}',
    '{"candidate_id":"C3","behavior":[]}',
    '{"candidate_id":"C3","candidate_id":"C4","behavior":"FOLLOW"}',
    '{"candidate_id":"C3","behavior":"REJOIN"} {"candidate_id":"C4","behavior":"FOLLOW"}',
])
def test_reject_ambiguous_or_uncontracted_actions(raw):
    with pytest.raises(ValueError):
        parse_decision(raw)


def test_task_plan_preserves_order_and_accepts_multiple_visits():
    plan = dict(steps=[dict(action='VISIT', direction='LEFT', distance_m=2),
                      dict(action='RETURN_HOME'),
                      dict(action='VISIT', direction='FORWARD', distance_m=1.5),
                      dict(action='STOP')])
    assert parse_task_plan('```json\n'+json.dumps(plan)+'\n```') == plan
    assert parse_task_plan('{"steps":[{"action":"STOP"}]}') == dict(steps=[dict(action='STOP')])


@pytest.mark.parametrize('plan', [
    {}, {'steps': []}, {'steps': 'STOP'},
    {'steps': [{'action': 'STOP'}], 'explanation': 'done'},
    {'steps': [{'action': 'VISIT', 'direction': 'LEFT', 'distance_m': 2.}]},
    {'steps': [{'action': 'STOP'}, {'action': 'RETURN_HOME'}, {'action': 'STOP'}]},
    {'steps': [{'action': 'RETURN_HOME', 'direction': 'LEFT'}, {'action': 'STOP'}]},
    {'steps': [{'action': 'STOP', 'candidate_id': 'HOLD'}]},
    {'steps': [{'action': 'EXECUTE', 'code': '__import__("os")'}, {'action': 'STOP'}]},
    {'steps': [{'action': 'VISIT', 'direction': 'UP', 'distance_m': 2.}, {'action': 'STOP'}]},
    {'steps': [{'action': 'VISIT', 'direction': ['LEFT'], 'distance_m': 2.}, {'action': 'STOP'}]},
    {'steps': [{'action': 'VISIT', 'direction': 'LEFT', 'distance_m': 2., 'extra': 0}, {'action': 'STOP'}]},
    {'steps': [None, {'action': 'STOP'}]},
])
def test_task_plan_rejects_unsupported_structure(plan):
    with pytest.raises(ValueError):
        parse_task_plan(json.dumps(plan))


@pytest.mark.parametrize('distance', [0., -1., float('nan'), float('inf'), '2.0', True, None])
def test_task_plan_rejects_nonfinite_nonpositive_or_non_numeric_distance(distance):
    with pytest.raises(ValueError):
        parse_task_plan(json.dumps(dict(steps=[dict(action='VISIT', direction='LEFT', distance_m=distance),
                                               dict(action='STOP')])))


def test_task_plan_bounds_action_count_and_rejects_multiple_objects():
    visit = dict(action='VISIT', direction='LEFT', distance_m=2.)
    assert len(parse_task_plan(json.dumps(dict(steps=[visit]*7+[dict(action='STOP')])))['steps']) == 8
    with pytest.raises(ValueError, match='eight'):
        parse_task_plan(json.dumps(dict(steps=[visit]*8+[dict(action='STOP')])))
    with pytest.raises(ValueError):
        parse_task_plan('{"steps":[{"action":"STOP"}]} {"steps":[{"action":"STOP"}]}')


def test_forest_site_plan_preserves_order_and_allows_revisits():
    plan = dict(steps=[dict(action='VISIT_SITE', site_id=site) for site in
                       ('ROW1_END', 'HEADLAND_1', 'ROW2_START', 'ROW2_END', 'HEADLAND_1', 'HOME')]
                     + [dict(action='STOP')])
    assert parse_forest_task_plan('```json\n'+json.dumps(plan)+'\n```') == plan
    assert parse_forest_task_plan('{"steps":[{"action":"STOP"}]}') == dict(steps=[dict(action='STOP')])


@pytest.mark.parametrize('site', ['', 'A'*65, '../HOME', '/tmp/HOME', 'ROW 1', 'ROW\n1',
                                  'HOME;exit', 'A.B', '__import__("os")', None, True, 1])
def test_forest_site_plan_rejects_unsafe_or_invalid_site_identifiers(site):
    with pytest.raises(ValueError, match='site_id'):
        parse_forest_task_plan(json.dumps(dict(steps=[dict(action='VISIT_SITE', site_id=site),
                                                    dict(action='STOP')])))


@pytest.mark.parametrize('plan', [
    {}, {'steps': []}, {'steps': 'HOME'},
    {'steps': [{'action': 'STOP'}], 'extra': 'unexpected'},
    {'steps': [{'action': 'VISIT_SITE', 'site_id': 'HOME'}]},
    {'steps': [{'action': 'STOP'}, {'action': 'VISIT_SITE', 'site_id': 'HOME'}, {'action': 'STOP'}]},
    {'steps': [{'action': 'VISIT_SITE', 'site_id': 'HOME', 'xyz': [0, 0, 2]}, {'action': 'STOP'}]},
    {'steps': [{'action': 'STOP', 'site_id': 'HOME'}]},
    {'steps': [{'action': 'RETURN_HOME'}, {'action': 'STOP'}]},
    {'steps': [{'action': 'VISIT', 'direction': 'LEFT', 'distance_m': 2.}, {'action': 'STOP'}]},
])
def test_forest_site_plan_rejects_non_site_schema_and_extra_fields(plan):
    with pytest.raises(ValueError):
        parse_forest_task_plan(json.dumps(plan))


def test_forest_site_plan_bounds_count_and_does_not_guess_catalog_membership():
    visit = dict(action='VISIT_SITE', site_id='A'*64)
    assert len(parse_forest_task_plan(json.dumps(dict(steps=[visit]*23+[dict(action='STOP')])))['steps']) == 24
    with pytest.raises(ValueError, match='24'):
        parse_forest_task_plan(json.dumps(dict(steps=[visit]*24+[dict(action='STOP')])))
    # ID membership remains a separate public-catalog check, never an oracle
    # hidden inside a syntax parser.
    assert parse_forest_task_plan('{"steps":[{"action":"VISIT_SITE","site_id":"Unregistered"},{"action":"STOP"}]}')['steps'][0]['site_id'] == 'Unregistered'


def worker(tmp_path, behavior='reply'):
    script = tmp_path/'worker.py'
    script.write_text('''import json,sys,time,os
print(json.dumps(dict(metadata=dict(fake=True))),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if 'request_file' in r:
  with open(r['request_file']) as f:r=json.load(f)
 MODE=MODE_VALUE
 if MODE=='hang': time.sleep(60)
 elif MODE=='die': sys.exit(7)
 elif MODE=='partial':
  sys.stdout.write('{"request_id":');sys.stdout.flush();time.sleep(60)
 else:
  raw='{"candidate_id":"C1","behavior":"FOLLOW"}'
  if r.get('response_schema')=='task_plan' and MODE!='wrong_plan':
   raw=json.dumps(dict(steps=[dict(action='VISIT',direction='LEFT',distance_m=2.),dict(action='RETURN_HOME'),dict(action='STOP')]))
  if r.get('response_schema')=='forest_task_plan' and MODE!='wrong_plan':
   raw=json.dumps(dict(steps=[dict(action='VISIT_SITE',site_id='ROW1_END'),dict(action='VISIT_SITE',site_id='HOME'),dict(action='STOP')]))
  print(json.dumps(dict(request_id=r['request_id'],raw=raw,latency_seconds=.01,input_tokens=10,output_tokens=5)),flush=True)
'''.replace('MODE_VALUE', repr(behavior)))
    image = tmp_path/'image.png'
    image.write_bytes(b'fake image; worker does not read it')
    return script, image


def result(client, deadline=2.):
    end = time.perf_counter()+deadline
    while time.perf_counter() < end:
        reply = client.poll()
        if reply is not None:
            return reply
        time.sleep(.005)
    raise AssertionError('no model result within test deadline')


def test_one_request_bounded_queue_and_repeatable_client(tmp_path):
    script, image = worker(tmp_path)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script) as client:
        assert client.metadata['fake'] and client.alive
        assert client.submit(image, 'Choose C1.', request_id='one')
        assert not client.submit(image, 'Ignored; no unbounded queue.', request_id='two')
        reply = result(client)
        assert reply['request_id'] == 'one'
        assert reply['decision'] == dict(candidate_id='C1', behavior='FOLLOW')
        assert reply['input_tokens'] == 10
        assert not client.busy and client.poll() is None
        assert client.submit(image, 'Choose C1.', request_id='two')
        assert result(client)['request_id'] == 'two'
    assert not client.alive
    client.close()


def test_text_only_task_plan_then_visual_decision_use_request_specific_schemas(tmp_path):
    script, image = worker(tmp_path)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script) as client:
        assert client.submit(prompt='先向左两米，然后返回起点并停止。', request_id='plan', response_schema='task_plan')
        assert not client.submit(image, 'Wait for plan completion.', request_id='visual')
        reply = result(client)
        assert reply['request_id'] == 'plan' and 'decision' not in reply
        assert [step['action'] for step in reply['task_plan']['steps']] == ['VISIT', 'RETURN_HOME', 'STOP']
        artifact = json.loads((tmp_path/'out/model_requests/000001.json').read_text())
        assert artifact['image'] is None and artifact['response_schema'] == 'task_plan'
        assert client.submit(image, 'Choose C1.', request_id='visual')
        assert result(client)['decision']['candidate_id'] == 'C1'


def test_wrong_plan_response_is_explicit_parse_error_not_decision(tmp_path):
    script, image = worker(tmp_path, 'wrong_plan')
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script) as client:
        assert client.submit(None, 'Plan the instruction.', response_schema='task_plan')
        reply = result(client)
        assert reply['task_plan'] is None and reply['error'].startswith('MODEL_OUTPUT:')
        assert 'decision' not in reply and client.alive


def test_plan_timeout_returns_plan_error_without_blocking_or_fabricated_steps(tmp_path):
    script, image = worker(tmp_path, 'hang')
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script,
                             inference_timeout=.1) as client:
        client.submit(None, 'Plan the instruction.', response_schema='task_plan')
        reply = result(client)
        assert reply['error'] == 'MODEL_TIMEOUT' and reply['task_plan'] is None
        assert 'decision' not in reply


def test_forest_plan_then_old_direction_plan_then_decision_keep_separate_results(tmp_path):
    script, image = worker(tmp_path)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script) as client:
        assert client.submit(None, '巡检一号行末，然后返回HOME，结束。', response_schema='forest_task_plan')
        reply = result(client)
        assert reply['forest_task_plan']['steps'][0] == dict(action='VISIT_SITE', site_id='ROW1_END')
        assert 'task_plan' not in reply and 'decision' not in reply
        assert client.submit(None, '向左两米后返回结束。', response_schema='task_plan')
        reply = result(client)
        assert reply['task_plan']['steps'][0]['direction'] == 'LEFT' and 'forest_task_plan' not in reply
        assert client.submit(image, 'Choose C1.')
        assert result(client)['decision']['candidate_id'] == 'C1'


@pytest.mark.parametrize('mode,error_prefix', [('wrong_plan', 'MODEL_OUTPUT:'), ('hang', 'MODEL_TIMEOUT')])
def test_forest_plan_errors_cannot_be_misread_as_a_movement_decision(tmp_path, mode, error_prefix):
    script, image = worker(tmp_path, mode)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script,
                             inference_timeout=.1) as client:
        assert client.submit(None, '巡检后返回。', response_schema='forest_task_plan')
        reply = result(client)
        assert reply['error'].startswith(error_prefix) and reply['forest_task_plan'] is None
        assert 'task_plan' not in reply and 'decision' not in reply


def test_request_schema_and_image_requirements(tmp_path):
    script, image = worker(tmp_path)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script) as client:
        with pytest.raises(ValueError, match='response_schema'):
            client.submit(image, 'Choose.', response_schema='code')
        with pytest.raises(ValueError, match='observation image'):
            client.submit(None, 'Choose.')
        with pytest.raises(ValueError, match='text-only'):
            client.submit(image, 'Plan.', response_schema='task_plan')
        with pytest.raises(ValueError, match='text-only'):
            client.submit(image, 'Plan forest.', response_schema='forest_task_plan')
        assert client.alive and not client.busy


@pytest.mark.parametrize('mode,expected', [('hang', 'MODEL_TIMEOUT'), ('partial', 'MODEL_TIMEOUT'), ('die', 'WORKER_EXIT: 7')])
def test_timeouts_partial_line_and_worker_exit_do_not_block_poll(tmp_path, mode, expected):
    script, image = worker(tmp_path, mode)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script,
                             inference_timeout=.1) as client:
        client.submit(image, 'Choose C1.', request_id='flight_17')
        began = time.perf_counter()
        reply = result(client)
        assert time.perf_counter()-began < 1.
        assert reply['error'] == expected
        assert reply['request_id'] == 'flight_17' and reply['decision'] is None
        assert client.poll() is None and not client.alive
        with pytest.raises(RuntimeError, match='not available'):
            client.submit(image, 'Choose C1.')


def test_startup_timeout_reaps_worker(tmp_path):
    script = tmp_path/'worker.py'
    script.write_text('import time;time.sleep(60)\n')
    began = time.perf_counter()
    with pytest.raises(TimeoutError, match='load timeout'):
        SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script, load_timeout=.05)
    assert time.perf_counter()-began < 1.


def test_large_prompt_is_spooled_and_excessive_prompt_rejected_before_sending(tmp_path):
    script, image = worker(tmp_path)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script) as client:
        with pytest.raises(ValueError, match='artifact budget'):
            client.submit(image, 'x'*300000)
        assert not client.busy and client.alive
        assert client.submit(image, 'Choose C1. '+'x'*10000)
        assert result(client)['decision']['candidate_id'] == 'C1'
        request = json.loads(next((tmp_path/'out/model_requests').glob('*.json')).read_text())
        assert len(request['prompt']) > 10000


class _LetterTokenizer:
    """Pieces as tokens, one token per capital letter, like Qwen's tokenizer."""
    def __init__(self, pieces):
        self.vocab = sorted(set(pieces) | {chr(c) for c in range(ord('A'), ord('Z')+1)})

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.index(text)]

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)


def test_letter_probabilities_are_read_at_the_choice_letter_only():
    import importlib.util
    import torch
    path = Path(__file__).resolve().parents[2]/'scripts/forest_subgoal_model_worker.py'
    spec = importlib.util.spec_from_file_location('forest_subgoal_model_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pieces = ['{"', 'next', '_', 'ais', 'le', '":', ' null', ',', ' "', 'choice', '":', ' "', 'B', '"}']
    tokenizer = _LetterTokenizer(pieces)
    generated = [tokenizer.vocab.index(p) for p in pieces]
    logits = [torch.zeros(1, len(tokenizer.vocab)) for _ in pieces]
    logits[12][0, tokenizer.vocab.index('B')], logits[12][0, tokenizer.vocab.index('A')] = 3., 2.
    letters = module.choice_letter_probabilities(tokenizer, generated, logits)
    assert letters['B'] > letters['A'] > letters['C'] > 0. and set(letters) == set(module.LETTERS)
    assert module.choice_letter_probabilities(tokenizer, generated[:12], logits) is None


def test_letter_scores_are_requested_only_for_a_choice(tmp_path):
    script, image = worker(tmp_path)
    with SubgoalModelProcess(sys.executable, tmp_path, tmp_path/'out', worker_script=script) as client:
        with pytest.raises(ValueError, match='choice schema'):
            client.submit(image, 'Choose C1.', letter_scores=True)
        assert client.submit(image, 'Choose C1.', request_id='plain')
        result(client)
        assert 'letter_scores' not in json.loads((tmp_path/'out/model_requests/000001.json').read_text())
