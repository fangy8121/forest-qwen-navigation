#!/usr/bin/env python3
"""Local pretrained Qwen3.5 image + candidate prompt -> short JSON response.

stdout is exclusively a line-delimited JSON protocol. Model/logging output goes
to stderr. No adapter, training state or prior conversation is loaded.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LETTERS = [chr(code) for code in range(ord('A'), ord('Z')+1)]


def choice_letter_probabilities(tokenizer, generated, logits):
    """Softmax probability of each letter A-Z at the step that wrote the choice letter.

    Each capital letter is one Qwen token; the step is the first one-letter
    token after ``"choice":"``. None if the reply has no such step.
    """
    import re
    import torch
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError('letter scoring needs one token per capital letter')
    text = ''
    for step, token in enumerate(generated):
        piece = tokenizer.decode([token])
        if piece in LETTERS and re.search(r'"choice"\s*:\s*"$', text):
            probability = torch.softmax(logits[step][0].float(), -1)[[ids[0] for ids in letters]]
            return {letter: float(p) for letter, p in zip(LETTERS, probability.tolist())}
        text += piece
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--image-max-pixels', type=int, default=262144)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            import torch
            from PIL import Image, ImageOps
            from omni_drones.navigation.forest.qwen import load_model
            from omni_drones.navigation.forest.qwen_subgoal import (
                PROTOCOL, TASK_PLAN_PROTOCOL, FOREST_TASK_PLAN_PROTOCOL, COMMAND_PROTOCOL)
            model, processor, metadata = load_model(args.model, args.device)
            model.eval()
            context = int(model.config.text_config.max_position_embeddings)
            if not 0 < args.max_new_tokens < args.max_input_tokens or args.image_max_pixels < 65536:
                raise ValueError('invalid token or image budget')
            if args.max_input_tokens+args.max_new_tokens > context:
                raise ValueError('input + output token budget exceeds model context')
            metadata.update(protocol=PROTOCOL, adapter=None, enable_thinking=False,
                            response_schemas=dict(decision=PROTOCOL, task_plan=TASK_PLAN_PROTOCOL,
                                                  forest_task_plan=FOREST_TASK_PLAN_PROTOCOL,
                                                  coverage_command=COMMAND_PROTOCOL),
                            max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                            image_max_pixels=args.image_max_pixels, do_sample=False,
                            timing='image read + processor + transfer + generate + decode; synchronized CUDA; excludes loading/rendering')
        print(json.dumps(dict(metadata=metadata), ensure_ascii=False), flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(dict(error=f'{type(error).__name__}: {error}')), flush=True)
        return
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if isinstance(request, dict) and set(request) == {'request_file'}:
                payload = Path(request['request_file']).read_bytes()
                if len(payload) > 262144:
                    raise ValueError('request artifact exceeds 262144 bytes')
                request = json.loads(payload)
            with redirect_stdout(sys.stderr):
                if str(args.device).startswith('cuda'):
                    torch.cuda.synchronize(args.device)
                began = time.perf_counter()
                schema = request.get('response_schema', 'decision')
                if schema == 'coverage_command':
                    if request.get('image') is not None:
                        raise ValueError('coverage_command is text-only')
                    rgb = None
                    system = (
                        'Interpret an online forest inspection command as a task specification, using ONLY the supplied observed memory. '
                        'Return one JSON with exactly intent, target, order, corridors, resume_after. '
                        'intent is COVER, PAUSE, RESUME, RETURN_HOME, VISIT_CORRIDOR or REVISIT_CORRIDOR. '
                        'target is null except VISIT_CORRIDOR/REVISIT_CORRIDOR. A target is one of '
                        '{"corridor_id":"<supplied ID>"}, {"lateral_index":k} (k-th aisle in the supplied lateral order, 1-based), '
                        'or {"relative":"left"|"right","count":n} (n aisles to the left/right of the current heading). '
                        'REVISIT_CORRIDOR only uses a supplied, previously entered corridor_id. '
                        'order is S or free for COVER, else null: S means adjacent aisles one by one; free means any order. '
                        'corridors is null, or for COVER a list of lateral indices when only some aisles are requested. '
                        'resume_after is true only if a temporary visit/return is followed by continuing the previous inspection. '
                        'count is how many aisles over from the current one: 相邻/旁边/右边那条/左边那条 = 1; '
                        '隔一行/隔一条 means ONE aisle lies in between, so count = 2; 隔两行 = 3; 隔N行 = N+1. '
                        'corridors must be null for 全部/整个/每一条/every/all aisles: never list the observed indices, '
                        'more aisles may still be discovered. '
                        'Interpret the CURRENT command, preserving temporal order. Examples: '
                        '巡检整个树林，完成后返回起点 -> {"intent":"COVER","target":null,"order":"S","corridors":null,"resume_after":false}. '
                        '只巡第1、3、5条通道 -> {"intent":"COVER","target":null,"order":"free","corridors":[1,3,5],"resume_after":false}. '
                        '先去右边隔一行的那条巡一遍，再回来继续 -> {"intent":"VISIT_CORRIDOR","target":{"relative":"right","count":2},"order":null,"corridors":null,"resume_after":true}. '
                        '去第4条通道巡检 -> {"intent":"VISIT_CORRIDOR","target":{"lateral_index":4},"order":null,"corridors":null,"resume_after":false}. '
                        '先返回起点，再继续巡检 -> {"intent":"RETURN_HOME","target":null,"order":null,"corridors":null,"resume_after":true}. '
                        '暂停 -> PAUSE. 继续 -> RESUME. 返航/返回起点结束 -> RETURN_HOME, resume_after false. '
                        'Returning AFTER coverage is the terminal policy, not an immediate RETURN_HOME. '
                        'Use the last departed corridor for 刚才那条/上一条. Never invent a corridor ID.')
                    content = [dict(type='text', text=request['prompt'])]
                elif schema == 'forest_task_plan':
                    if request.get('image') is not None:
                        raise ValueError('forest_task_plan is text-only')
                    rgb = None
                    system = (
                        'Compile the user forest inspection instruction using the supplied public site catalog. '
                        'Preserve all requested sites and their visit order, including returning HOME when requested. '
                        'Only reference site IDs provided in that catalog; do not invent IDs or coordinates. '
                        'Return only one JSON object with exactly steps, a list of 1 to 24 action objects. '
                        'The only allowed steps are {"action":"VISIT_SITE","site_id":"catalog_id"} '
                        'and {"action":"STOP"}. Use exactly those keys. '
                        'Each VISIT_SITE is a task destination; actual online obstacle avoidance is handled separately. '
                        'Every visit implies measured arrival; do not add STOP steps between sites. '
                        'STOP is mandatory and must occur only as the final step. '
                        'Do not add unrequested visits. Output no explanations, markdown, executable code or reasoning.')
                    content = [dict(type='text', text=request['prompt'])]
                elif schema == 'task_plan':
                    if request.get('image') is not None:
                        raise ValueError('task_plan is text-only')
                    rgb = None
                    system = (
                        'Compile the user drone instruction into an ordered task plan. '
                        'Preserve all requested directions, distances and their order. '
                        'Directions are relative to the recorded takeoff heading. '
                        'Return only one JSON object with exactly the key steps, a list of 1 to 8 action objects. '
                        'Allowed steps are {"action":"VISIT","direction":"LEFT|RIGHT|FORWARD|BACKWARD",'
                        '"distance_m":positive_number}, {"action":"RETURN_HOME"}, {"action":"STOP"}. '
                        'RETURN_HOME means go back to the recorded takeoff position. '
                        'STOP is mandatory and must occur only as the final step. '
                        'Use only those actions and exact keys. Do not invent extra destinations or return home '
                        'unless the user requests it. Do not output candidates, coordinates, code, reasoning or markdown.')
                    content = [dict(type='text', text=request['prompt'])]
                elif schema == 'choice':
                    rgb = []
                    for item in request.get('images') or [request['image']]:
                        with Image.open(item) as source:
                            rgb.append(ImageOps.exif_transpose(source).convert('RGB'))
                    system = (
                        'You plan the next step of a forest inspection drone. Look at both images and the text, '
                        'follow the user instruction, and pick exactly one offered candidate by its letter. '
                        'Return only one JSON object with exactly next_aisle and choice, e.g. '
                        '{"next_aisle":2,"choice":"B"}.')
                    content = [*(dict(type='image', image=item) for item in rgb),
                               dict(type='text', text=request['prompt'])]
                elif schema == 'decision':
                    with Image.open(request['image']) as source:
                        rgb = ImageOps.exif_transpose(source).convert('RGB')
                    system = (
                        'You select one observed candidate subgoal for a drone. '
                        'Follow the user task and use the supplied observations and measured history. '
                        'Return only one JSON object with exactly candidate_id and behavior. '
                        'Do not invent candidate IDs or coordinates.')
                    content = [dict(type='image', image=rgb), dict(type='text', text=request['prompt'])]
                else:
                    raise ValueError('unknown response_schema')
                messages = [dict(role='system', content=system), dict(role='user', content=content)]
                prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
                image_kwargs = (dict(images=rgb if isinstance(rgb, list) else [rgb], images_kwargs=dict(size=dict(
                    shortest_edge=65536, longest_edge=args.image_max_pixels))) if rgb is not None else {})
                inputs = processor(text=[prefix], return_tensors='pt', add_special_tokens=False,
                                   padding=False, truncation=False, **image_kwargs)
                input_tokens = int(inputs['input_ids'].shape[-1])
                if input_tokens > args.max_input_tokens:
                    raise ValueError(f'input has {input_tokens} tokens; exceeds {args.max_input_tokens}; no truncation applied')
                scored = bool(request.get('letter_scores')) and schema == 'choice'
                with torch.inference_mode():
                    ids = model.generate(**inputs.to(args.device), max_new_tokens=args.max_new_tokens,
                                         do_sample=False, use_cache=True,
                                         pad_token_id=processor.tokenizer.eos_token_id,
                                         **(dict(output_logits=True, return_dict_in_generate=True) if scored else {}))
                logits = ids.logits if scored else None
                ids = ids.sequences if scored else ids
                generated = ids[:, input_tokens:]
                raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
                if str(args.device).startswith('cuda'):
                    torch.cuda.synchronize(args.device)
                result = dict(request_id=request['request_id'], raw=raw,
                              response_schema=schema,
                              latency_seconds=time.perf_counter()-began, input_tokens=input_tokens,
                              output_tokens=int(generated.shape[-1]),
                              output_limit_reached=bool(generated.shape[-1] >= args.max_new_tokens))
                if scored:
                    result['letter_probabilities'] = choice_letter_probabilities(
                        processor.tokenizer, generated[0].tolist(), logits)
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = dict(request_id=request.get('request_id') if isinstance(request, dict) else None,
                          error=f'{type(error).__name__}: {error}')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
