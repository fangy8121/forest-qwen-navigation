"""Zero-shot candidate decisions in an isolated, bounded Qwen worker.

This module deliberately imports neither Torch nor Isaac. Loading is completed
before a flight; submitting and polling never wait for model inference. The
supervisor remains responsible for candidate identity, task versions and fresh
geometry. A syntactically valid model response is not an execution permission.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import time


BEHAVIORS = frozenset({
    'FOLLOW', 'CHANGE_AISLE', 'REJOIN', 'OBSERVE', 'BACKTRACK', 'RETURN', 'REQUEST_STOP',
})
PROTOCOL = 'forest_candidate_subgoal_v1'
TASK_PLAN_PROTOCOL = 'forest_directional_task_plan_v1'
FOREST_TASK_PLAN_PROTOCOL = 'forest_public_site_task_plan_v1'
COMMAND_PROTOCOL = 'forest_runtime_command_v2'


COMMAND_INTENTS = ('COVER', 'PAUSE', 'RESUME', 'RETURN_HOME', 'REVISIT_CORRIDOR', 'VISIT_CORRIDOR')


def parse_coverage_command(raw):
    """One runtime command as a task specification; v1 objects remain valid.

    v2 keys: intent, target, order, corridors, resume_after. ``target`` names a
    corridor by observed ID, by 1-based lateral index, or relative to the
    current heading (left/right, count). ``order`` (S or free) and the
    ``corridors`` subset (lateral indices) only apply to COVER. The result is
    normalized to all v2 keys plus the v1 ``corridor_id``.
    """
    obj = _one_object(raw)
    if set(obj) == {'intent', 'corridor_id', 'resume_after'}:
        obj = dict(intent=obj['intent'], resume_after=obj['resume_after'], order=None, corridors=None,
                   target=None if obj['corridor_id'] is None else dict(corridor_id=obj['corridor_id']))
        version = 1
    elif set(obj) == {'intent', 'target', 'order', 'corridors', 'resume_after'}:
        version = 2
    else:
        raise ValueError('command requires intent, target, order, corridors, resume_after')
    if obj['intent'] not in (COMMAND_INTENTS if version == 2 else COMMAND_INTENTS[:-1]):
        raise ValueError('unsupported coverage command intent')
    if not isinstance(obj['resume_after'], bool):
        raise ValueError('resume_after must be boolean')
    target = obj['target']
    if target is not None:
        if not isinstance(target, dict) or len(target) not in (1, 2):
            raise ValueError('target must be one corridor reference or null')
        if set(target) == {'corridor_id'}:
            if not isinstance(target['corridor_id'], str) or not re.fullmatch(r'R\d+_R\d+', target['corridor_id']):
                raise ValueError('corridor_id must be an observed corridor ID')
        elif set(target) == {'lateral_index'}:
            if type(target['lateral_index']) is not int or target['lateral_index'] < 1:
                raise ValueError('lateral_index must be a positive integer')
        elif set(target) == {'relative', 'count'}:
            if target['relative'] not in ('left', 'right') or type(target['count']) is not int or target['count'] < 1:
                raise ValueError('relative target needs left/right and a positive count')
        else:
            raise ValueError('target uses corridor_id, lateral_index or relative+count')
    if (obj['intent'] in ('REVISIT_CORRIDOR', 'VISIT_CORRIDOR')) != (target is not None):
        raise ValueError('only VISIT_CORRIDOR/REVISIT_CORRIDOR require a target')
    if obj['intent'] == 'REVISIT_CORRIDOR' and set(target) != {'corridor_id'}:
        raise ValueError('REVISIT_CORRIDOR names a previously entered corridor ID')
    if obj['intent'] == 'COVER':
        if obj['order'] not in (None, 'S', 'free'):
            raise ValueError('order must be S, free or null')
        corridors = obj['corridors']
        if corridors is not None and (not isinstance(corridors, list) or not corridors
                or any(type(i) is not int or i < 1 for i in corridors) or len(set(corridors)) != len(corridors)):
            raise ValueError('corridors must be distinct positive lateral indices or null')
    else:
        # Meaningless outside COVER; dropping them cannot widen any permission.
        obj['order'] = obj['corridors'] = None
    if obj['intent'] in ('PAUSE', 'RESUME', 'COVER') and obj['resume_after']:
        raise ValueError('resume_after only applies to a temporary visit/return')
    obj['corridor_id'] = (target or {}).get('corridor_id')
    return obj


def _one_object(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('model response must contain one JSON object')

    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f'duplicate JSON key: {key}')
            value[key] = item
        return value

    decoder, objects, end = json.JSONDecoder(object_pairs_hook=unique_keys), [], 0
    for index, char in enumerate(raw):
        if char != '{' or index < end:
            continue
        try:
            obj, length = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        objects.append(obj)
        end = index+length
    if len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError('model response must contain exactly one JSON object')
    return objects[0]


def parse_decision(raw):
    """Accept one JSON object, including a Markdown fence, with exact fields.

    A short explanatory prefix/suffix is retained in ``raw`` by the caller but
    may surround the object. Multiple objects, additional fields and malformed
    decisions are rejected instead of guessing which command was intended.
    Candidate membership is checked by the task supervisor against its snapshot.
    """
    obj = _one_object(raw)
    if set(obj) != {'candidate_id', 'behavior'}:
        raise ValueError('decision requires exactly candidate_id and behavior')
    if (not isinstance(obj['candidate_id'], str) or not obj['candidate_id'].strip()
            or len(obj['candidate_id']) > 128):
        raise ValueError('candidate_id must be a nonempty string of at most 128 characters')
    if not isinstance(obj['behavior'], str) or obj['behavior'] not in BEHAVIORS:
        raise ValueError('unsupported decision behavior')
    return obj


def parse_choice(raw):
    """One JSON object with exactly ``next_aisle`` and ``choice`` (one letter).

    Stating the aisle first halved the letter-position errors of the base 4B
    model offline (7/9 instead of 4/9). Only the letter selects; the aisle is
    logged. Whether the letter names an offered candidate is checked later.
    """
    obj = _one_object(raw)
    if set(obj) != {'next_aisle', 'choice'}:
        raise ValueError('choice requires exactly next_aisle and choice')
    if not isinstance(obj['choice'], str) or len(obj['choice'].strip()) != 1 or not obj['choice'].strip().isupper():
        raise ValueError('choice must be one uppercase candidate letter')
    aisle = obj['next_aisle']
    # The aisle is only logged, so its spelling must not void a valid letter
    # (run c65a882d: five return choices with "null" ended the run).
    if isinstance(aisle, str) and aisle.strip().lower() == 'null':
        aisle = None
    elif isinstance(aisle, str) and aisle.strip().isdigit():
        aisle = int(aisle.strip())
    if aisle is not None and (type(aisle) is not int or aisle < 1):
        raise ValueError('next_aisle must be a positive integer or null')
    return dict(next_aisle=aisle, choice=obj['choice'].strip())


def parse_task_plan(raw):
    """Validate language-compiled steps; never interpret code or infer a task.

    The model supplies order and direction. The deterministic supervisor owns
    measured completion and execution feasibility; this parser only validates
    the finite supported language-task protocol. STOP is permitted only last.
    """
    obj = _one_object(raw)
    if set(obj) != {'steps'} or not isinstance(obj['steps'], list) or not 1 <= len(obj['steps']) <= 8:
        raise ValueError('task plan requires exactly steps, containing one to eight steps')
    steps = []
    for i, step in enumerate(obj['steps']):
        if not isinstance(step, dict) or not isinstance(step.get('action'), str):
            raise ValueError('each task step must be an action object')
        action = step['action']
        if action == 'VISIT':
            if set(step) != {'action', 'direction', 'distance_m'}:
                raise ValueError('VISIT requires exactly action, direction and distance_m')
            direction = step['direction']
            if not isinstance(direction, str) or direction not in ('LEFT', 'RIGHT', 'FORWARD', 'BACKWARD'):
                raise ValueError('unsupported VISIT direction')
            distance = _positive(step['distance_m'], 'VISIT distance_m')
            steps.append(dict(action=action, direction=direction, distance_m=distance))
        elif action in ('RETURN_HOME', 'STOP'):
            if set(step) != {'action'}:
                raise ValueError(f'{action} requires exactly action')
            if action == 'STOP' and i != len(obj['steps'])-1:
                raise ValueError('STOP must be the final task step')
            steps.append(dict(action=action))
        else:
            raise ValueError('unsupported task action')
    if steps[-1]['action'] != 'STOP':
        raise ValueError('the final task step must be STOP')
    return dict(steps=steps)


def parse_forest_task_plan(raw):
    """Validate a public-site visit sequence without inventing catalog entries.

    Catalog membership and task correctness belong to the runner/evaluator.
    IDs are short ASCII identifiers and cannot contain paths, whitespace or
    executable expressions. Repeated visits are valid (e.g. return to HOME).
    """
    obj = _one_object(raw)
    if set(obj) != {'steps'} or not isinstance(obj['steps'], list) or not 1 <= len(obj['steps']) <= 24:
        raise ValueError('forest task plan requires exactly steps, containing one to 24 steps')
    steps = []
    for index, step in enumerate(obj['steps']):
        if not isinstance(step, dict) or not isinstance(step.get('action'), str):
            raise ValueError('each forest task step must be an action object')
        if step['action'] == 'VISIT_SITE':
            if set(step) != {'action', 'site_id'}:
                raise ValueError('VISIT_SITE requires exactly action and site_id')
            site_id = step['site_id']
            if not isinstance(site_id, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', site_id) is None:
                raise ValueError('site_id must be a safe ASCII identifier of one to 64 characters')
            steps.append(dict(action='VISIT_SITE', site_id=site_id))
        elif step['action'] == 'STOP':
            if set(step) != {'action'}:
                raise ValueError('STOP requires exactly action')
            if index != len(obj['steps'])-1:
                raise ValueError('STOP must be the final forest task step')
            steps.append(dict(action='STOP'))
        else:
            raise ValueError('unsupported forest task action')
    if steps[-1]['action'] != 'STOP':
        raise ValueError('the final forest task step must be STOP')
    return dict(steps=steps)


def _positive(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


class SubgoalModelProcess:
    """One request in flight, bounded startup/inference, no background threads.

    ``submit`` writes a small JSON line to an idle worker and returns immediately.
    ``poll`` uses a nonblocking binary stdout pipe and returns ``None`` until a
    complete response is available. On timeout or process death it returns one
    error result, kills the worker, and refuses future submissions.
    """

    def __init__(self, python, model, output, *, device='cuda:0', max_input_tokens=4096,
                 max_new_tokens=96, image_max_pixels=262144, load_timeout=180.,
                 inference_timeout=20., worker_script=None):
        self.load_timeout = _positive(load_timeout, 'load_timeout')
        self.inference_timeout = _positive(inference_timeout, 'inference_timeout')
        if (not isinstance(max_input_tokens, int) or not isinstance(max_new_tokens, int)
                or not 0 < max_new_tokens < max_input_tokens):
            raise ValueError('require 0 < max_new_tokens < max_input_tokens')
        if not isinstance(image_max_pixels, int) or image_max_pixels < 65536:
            raise ValueError('image_max_pixels must be at least 65536')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self._output = output.resolve()
        self._log = (output/'model_worker.log').open('ab', buffering=0)
        self._requests = self._output/'model_requests'
        self._requests.mkdir(exist_ok=True)
        root = Path(__file__).resolve().parents[3]
        script = Path(worker_script) if worker_script is not None else root/'scripts/forest_subgoal_model_worker.py'
        command = [str(Path(python).expanduser().absolute()), str(script.resolve()),
                   '--model', str(Path(model).expanduser().resolve()), '--device', str(device),
                   '--max-input-tokens', str(max_input_tokens), '--max-new-tokens', str(max_new_tokens),
                   '--image-max-pixels', str(image_max_pixels)]
        self._closed, self._dead, self._pending = False, False, None
        self._buffer, self._request_count = b'', 0
        self.process = None
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self._log, bufsize=0)
            os.set_blocking(self.process.stdout.fileno(), False)
            reply = self._wait_startup(started+self.load_timeout)
            if set(reply) != {'metadata'} or not isinstance(reply['metadata'], dict):
                raise RuntimeError(f'invalid model startup reply: {reply}')
            self.metadata = dict(reply['metadata'], worker_python=command[0],
                                 worker_script=str(script.resolve()),
                                 worker_load_seconds=time.perf_counter()-started,
                                 inference_timeout_seconds=self.inference_timeout)
            (output/'model_metadata.json').write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2)+'\n')
        except BaseException:
            self.close()
            raise

    @property
    def busy(self):
        return self._pending is not None

    @property
    def alive(self):
        return not self._closed and not self._dead and self.process is not None and self.process.poll() is None

    def _line(self):
        """Drain available bytes only; a partial line never blocks the caller."""
        if b'\n' not in self._buffer:
            while True:
                try:
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                self._buffer += chunk
                if len(self._buffer) > 1048576:
                    raise RuntimeError('worker stdout exceeded the one-megabyte protocol bound')
                if b'\n' in self._buffer:
                    break
        if b'\n' not in self._buffer:
            return None
        line, self._buffer = self._buffer.split(b'\n', 1)
        try:
            data = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError('worker emitted invalid JSON') from error
        if not isinstance(data, dict):
            raise RuntimeError('worker emitted a non-object response')
        return data

    def _wait_startup(self, deadline):
        while True:
            reply = self._line()
            if reply is not None:
                if 'error' in reply:
                    raise RuntimeError(f'model loading failed: {reply["error"]}')
                return reply
            if self.process.poll() is not None:
                raise RuntimeError(f'model worker exited during startup ({self.process.returncode}); see model_worker.log')
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                raise TimeoutError('model worker load timeout; see model_worker.log')
            select.select([self.process.stdout], [], [], min(.1, remaining))

    def submit(self, image=None, prompt=None, *, request_id=None, response_schema='decision', letter_scores=False):
        """``letter_scores`` (choice only) also returns each letter's probability at the choice step."""
        if self._closed or self._dead or self.process.poll() is not None:
            raise RuntimeError('model worker is not available')
        if self.busy:
            return False
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('a nonempty prompt is required')
        if response_schema not in ('decision', 'choice', 'task_plan', 'forest_task_plan', 'coverage_command'):
            raise ValueError('unsupported response_schema')
        if letter_scores and response_schema != 'choice':
            raise ValueError('letter_scores needs the choice schema')
        path, paths = None, None
        if isinstance(image, (list, tuple)):
            # Several images (point_choice_v1: overview + local BEV), in order.
            paths = [Path(item).expanduser().resolve() for item in image]
            for item in paths:
                if not item.is_file():
                    raise FileNotFoundError(item)
            path = paths[0] if paths else None
        elif image is not None:
            path = Path(image).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
        if response_schema in ('decision', 'choice') and path is None:
            raise ValueError('candidate decision requires an observation image')
        if response_schema in ('task_plan', 'forest_task_plan', 'coverage_command') and path is not None:
            raise ValueError(f'{response_schema} compilation is text-only; image must be None')
        self._request_count += 1
        request_id = str(request_id if request_id is not None else f'qwen_{self._request_count:06d}')
        request = dict(request_id=request_id, image=str(path) if path is not None else None,
                       prompt=prompt, response_schema=response_schema)
        if paths is not None:
            request['images'] = [str(item) for item in paths]
        if letter_scores:
            request['letter_scores'] = True
        encoded_request = (json.dumps(request, ensure_ascii=False)+'\n').encode('utf-8')
        if len(encoded_request) > 262144:
            raise ValueError('worker request exceeds 262144-byte artifact budget; shorten prompt')
        artifact = self._requests/f'{self._request_count:06d}.json'
        # Retain the exact prompt for audit. Only its small path frame crosses
        # the pipe, so larger candidate tables cannot stall the control loop.
        artifact.write_bytes(encoded_request)
        encoded = (json.dumps(dict(request_file=str(artifact)))+'\n').encode('utf-8')
        # Linux PIPE_BUF is at least 4096; the sole idle worker is waiting for a
        # line. Enforce a bounded message that cannot fill its input pipe.
        pipe_bound = os.fpathconf(self.process.stdin.fileno(), 'PC_PIPE_BUF')
        if len(encoded) > pipe_bound:
            raise ValueError(f'worker artifact path exceeds {pipe_bound}-byte nonblocking pipe budget')
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            sent = os.write(self.process.stdin.fileno(), encoded)
        except (BlockingIOError, BrokenPipeError) as error:
            self._dead = True
            self._terminate(wait=False)
            raise RuntimeError('model worker input unavailable') from error
        if sent != len(encoded):
            self._dead = True
            self._terminate(wait=False)
            raise RuntimeError('partial worker request; worker terminated')
        self._pending = dict(request_id=request_id, started=time.perf_counter(), response_schema=response_schema)
        return True

    def _failure(self, error):
        pending, self._pending = self._pending, None
        self._dead = True
        self._terminate(wait=False)
        key = pending['response_schema']
        return dict(request_id=pending['request_id'], error=error,
                    wall_seconds=time.perf_counter()-pending['started'], raw=None,
                    **{key: None})

    def poll(self):
        if self._closed or self._pending is None:
            return None
        try:
            reply = self._line()
        except RuntimeError as error:
            return self._failure(f'WORKER_PROTOCOL: {error}')
        if reply is not None:
            if reply.get('request_id') != self._pending['request_id']:
                return self._failure('WORKER_REQUEST_ID')
            pending, self._pending = self._pending, None
            reply['wall_seconds'] = time.perf_counter()-pending['started']
            key = pending['response_schema']
            parser = dict(decision=parse_decision, choice=parse_choice, task_plan=parse_task_plan,
                          forest_task_plan=parse_forest_task_plan,
                          coverage_command=parse_coverage_command)[key]
            if 'error' not in reply:
                try:
                    reply[key] = parser(reply.get('raw'))
                except (ValueError, TypeError) as error:
                    reply.update(error=f'MODEL_OUTPUT: {error}', **{key: None})
            else:
                reply[key] = None
            return reply
        if self.process.poll() is not None:
            return self._failure(f'WORKER_EXIT: {self.process.returncode}')
        if time.perf_counter()-self._pending['started'] >= self.inference_timeout:
            return self._failure('MODEL_TIMEOUT')
        return None

    def _terminate(self, *, wait):
        if self.process is not None and self.process.poll() is None:
            if not wait:
                # ``poll`` is called by the physical loop. Never wait even for
                # process shutdown here; close() reaps it after flight cleanup.
                self.process.kill()
                return
            self.process.terminate()
            # Never join an inference thread. Kill after a short bounded grace.
            try:
                self.process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pending = None
        self._terminate(wait=True)
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
