"""Qwen3.5 主实验 / Qwen3-VL 对照、六点协议与答案损失。"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from .geometry import BOUNDS, HEIGHT_LEVELS, PROTOCOL_VERSION, Z_BOUNDS
from .observation import representation_spec


def task_prompt(goal, view='bev', corridor=None, velocity=None, acceleration=None, protocol_version=PROTOCOL_VERSION):
    if protocol_version != PROTOCOL_VERSION:
        raise ValueError('XYZ planning requires protocol 3; rebuild the dataset from raw scans')
    shape = '四边形' if corridor is not None and len(corridor) == 4 else '多边形'
    constraint=('紫色虚线为任务通道边界，不得跨出。通道'+shape+'XY='+str(np.round(corridor,3).tolist())+'。'
                if corridor is not None else '')
    motion = (f'当前局部速度XYZ={np.round(velocity,3).tolist()}m/s；'
              f'加速度XYZ={np.round(acceleration,3).tolist()}m/s²。'
              if velocity is not None and acceleration is not None else '')
    layers = f'{len(HEIGHT_LEVELS)}个等权俯视图' if view == 'bev' else f'{len(HEIGHT_LEVELS)}个等权高度平面'
    lower_view = ('右侧的Lower hits小图只显示主高度带以下的观测障碍，颜色表示相对z；'
                  '小图灰底及没有命中点的位置不表示自由空间。' if view == 'bev' else '')
    return ('无人机局部多高度观测：x前/y左/z上，单位米；所有坐标相对当前机体中心，保持重力竖直。'
            f'{layers}对应机体中心相对高度z={list(HEIGHT_LEVELS)}m，共用XY坐标。'
            '蓝色为当前位置及前向，绿色为目标，橙色为考虑机体体积和安全余量后的障碍区，'
            '灰色未知，浅色为已观测空间。'+lower_view+
            f'目标XYZ={np.round(goal,3).tolist()}；XY边界[xmin,xmax,ymin,ymax]={list(BOUNDS)}；'
            f'z范围={list(Z_BOUNDS)}。图示中心高度不限制输出高度；未展示的高度不能据图认定可通行。'
            '可选择范围内任意高度；层间升降的整段路径也须安全。'
            +constraint+motion+'目标被挡时可选择安全中间点绕行、上跨或下穿。'
            '整条未来路径的机体空间须已观测，并保有安全制动空间；HOLD表示悬停等待重观测。'
            '不要输出Markdown代码围栏、解释或额外字段。'
            '输出唯一JSON对象：{"status":"GO","points":[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],'
            '[x4,y4,z4],[x5,y5,z5],[x6,y6,z6]]}，恰好6个未来点；'
            '无法安全前进则{"status":"HOLD","points":[]}。')


def messages(prompt, answer=None):
    result = [{'role': 'user', 'content': [{'type':'image'}, {'type':'text','text':prompt}]}]
    if answer is not None:
        result.append({'role':'assistant', 'content':[{'type':'text','text':answer}]})
    return result


def assistant_labels(full_ids, prefix_ids, attention_mask=None):
    import torch
    n = prefix_ids.shape[-1]
    if full_ids.shape[-1] <= n or not torch.equal(full_ids[..., :n], prefix_ids):
        raise ValueError('chat template prefix mismatch or empty assistant answer')
    # 图像扩展 token 与用户提示全部为 -100；只监督答案和结束标记。
    labels = full_ids.clone()
    labels[..., :n] = -100
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    return labels


def encode_sample(processor, image, prompt, answer=None, max_tokens=4096,
                  coordinate_weight=1., maneuver_multiplier=1.):
    from PIL import Image
    with Image.open(image) as f:
        rgb = f.convert('RGB')
    prefix = processor.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_batch = processor(text=[prefix], images=[rgb], return_tensors='pt', padding=False, truncation=False)
    if answer is None:
        batch = prompt_batch
    else:
        text = processor.apply_chat_template(messages(prompt, answer), tokenize=False, add_generation_prompt=False, enable_thinking=False)
        batch = processor(text=[text], images=[rgb], return_tensors='pt', padding=False, truncation=False)
        batch['labels'] = assistant_labels(batch['input_ids'], prompt_batch['input_ids'], batch.get('attention_mask'))
        if coordinate_weight != 1. or maneuver_multiplier != 1.:
            from .supervision import answer_token_weights, parse_prompt_goal
            batch['token_weights'] = answer_token_weights(processor.tokenizer,
                batch['input_ids'], batch['labels'], answer, coordinate_weight, maneuver_multiplier,
                goal=parse_prompt_goal(prompt))
    if batch['input_ids'].shape[-1] > max_tokens:
        raise ValueError(f'sample exceeds {max_tokens} tokens; no truncation allowed: {image}')
    return batch


def load_model(base, device):
    import torch
    import transformers
    from transformers import AutoProcessor
    if base is None or not Path(base).expanduser().is_dir():
        raise FileNotFoundError(f'model directory missing: {base}; run scripts/download_forest_models.py')
    base = Path(base).expanduser().resolve()
    config = json.loads((base/'config.json').read_text())
    classes = {'qwen3_5': 'Qwen3_5ForConditionalGeneration', 'qwen3_vl': 'Qwen3VLForConditionalGeneration'}
    model_class = getattr(transformers, classes[config['model_type']])
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; repair environment or select CPU explicitly')
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    model = model_class.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32).to(device)
    metadata = dict(base_model=str(base), model_config=config,
        config_sha256=hashlib.sha256((base/'config.json').read_bytes()).hexdigest(),
        checkpoint_files={p.name: {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                          for p in sorted(base.glob('*.safetensors'))},
        transformers=importlib.metadata.version('transformers'), torch=torch.__version__)
    origin = base/'model_origin.json'
    if origin.is_file():
        metadata['source'] = json.loads(origin.read_text())
    return model, processor, metadata


def lora_targets(model):
    # Qwen3.5 同时覆盖 full attention 与 Gated DeltaNet 的线性投影。
    import torch
    projections = {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z',
                   'in_proj_b','in_proj_a','in_proj_qkvz','in_proj_ba','out_proj'}
    names = [name for name, module in model.named_modules()
             if '.language_model.' in name and isinstance(module, torch.nn.Linear)
             and name.rsplit('.',1)[-1] in projections]
    if not names:
        raise ValueError('no language attention projections found')
    return names


def visual_merger_targets(model):
    """Only visual-to-language mergers; encoder blocks remain frozen.

    Qwen3.5 uses the main merger. Qwen3-VL also has deep-stack mergers.
    Matching complete module paths avoids selecting similarly named layers.
    """
    import torch
    names = [name for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear)
             and ('.visual.merger.' in name or '.visual.deepstack_merger_list.' in name)]
    if not names:
        raise ValueError('no visual-to-language merger projections found')
    return names


def attach_training_lora(model, rank=8, adapter=None, vision_adapter=False):
    """Attach exact attention/optional merger targets, preserving a warm start.

    A language-only source may initialize the expanded adapter: new merger
    B matrices start at zero, so the initial model function is unchanged.
    Existing source projections must all load; only the new merger may be missing.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    language = lora_targets(model)
    targets = language + (visual_merger_targets(model) if vision_adapter else [])
    model.requires_grad_(False)
    expanded = None
    if adapter is not None:
        config = json.loads((Path(adapter)/'adapter_config.json').read_text())
        if (config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM'
                or config.get('r') != rank or config.get('lora_alpha') != 2*rank
                or config.get('lora_dropout') != .05 or config.get('bias') != 'none'
                or any(config.get(key) for key in ('modules_to_save','rank_pattern','alpha_pattern',
                       'use_rslora','use_dora','lora_bias','use_qalora','alora_invocation_tokens','loftq_config'))):
            raise ValueError('initial adapter LoRA settings do not match the requested attention/merger configuration')
        selectors = config.get('target_modules', [])
        if not isinstance(selectors, list) or any(not isinstance(value, str) for value in selectors):
            raise ValueError('initial adapter must list exact attention/merger target modules')
        expanded = {name for name,module in model.named_modules() if isinstance(module, torch.nn.Linear)
                    and any(name == selector or name.endswith('.'+selector) for selector in selectors)}
        if expanded != set(targets) and not (vision_adapter and expanded == set(language)):
            raise ValueError('initial adapter targets do not match the requested attention/merger projections')
    if adapter is not None and expanded == set(targets):
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=.05,
            target_modules=targets, bias='none', task_type='CAUSAL_LM'))
        if adapter is not None:
            weights = load_peft_weights(str(adapter), device='cpu')
            loaded = set_peft_model_state_dict(model, weights)
            missing = {name for name in loaded.missing_keys if 'lora_' in name}
            expected = {name for name,param in model.named_parameters()
                        if 'lora_' in name and '.visual.' in name}
            if missing != expected or loaded.unexpected_keys:
                raise ValueError('initial adapter weights failed to load exactly; only new merger weights may be absent')
            if any(torch.count_nonzero(param) for name,param in model.named_parameters()
                   if '.visual.' in name and '.lora_B.' in name):
                raise RuntimeError('new visual merger adapter must initially have zero output')
    injected = {name for name,module in model.base_model.model.named_modules()
                if hasattr(module, 'lora_A') and 'default' in module.lora_A}
    if injected != set(targets):
        raise ValueError('adapter did not inject exactly the requested attention/merger projections')
    allowed = tuple(target+'.lora_' for target in targets)
    if any(not name.removeprefix('base_model.model.').startswith(allowed)
           for name,param in model.named_parameters() if param.requires_grad):
        raise RuntimeError('parameters outside the requested LoRA projections are trainable')
    return model, targets


class QwenPlanner:
    def __init__(self, base, adapter=None, device='cuda:0', max_input_tokens=4096, max_new_tokens=256):
        if adapter is not None:
            if not Path(adapter).is_dir():
                raise FileNotFoundError(adapter)
            training = Path(adapter)/'training.json'
            manifest = json.loads(training.read_text()).get('manifest', {}) if training.is_file() else {}
            if (manifest.get('protocol_version') != PROTOCOL_VERSION or
                    manifest.get('representation') != representation_spec()):
                raise ValueError('adapter uses an incompatible observation/path protocol; train a new XYZ adapter')
        self.model, self.processor, self.metadata = load_model(base, device)
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, local_files_only=True)
        self.metadata.update(protocol_version=PROTOCOL_VERSION, representation=representation_spec())
        self.model.eval()
        self.device, self.max_input_tokens, self.max_new_tokens = device, max_input_tokens, max_new_tokens
        self.metadata.update(adapter=str(Path(adapter).resolve()) if adapter else None,
            timing='image load + processor + device transfer + generate + decode; excludes model load and rendering',
            max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, do_sample=False)

    def generate(self, image, prompt):
        import torch
        # CUDA 同步使墙钟耗时包含真实推理完成，而不是异步提交时间。
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        batch = encode_sample(self.processor, image, prompt, max_tokens=self.max_input_tokens).to(self.device)
        with torch.inference_mode():
            ids = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)
        return text, time.perf_counter()-start
