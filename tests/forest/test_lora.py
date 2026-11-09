"""可选依赖集成测试：随机小模型，不代表预训练 Qwen 的规划效果。"""
import pytest
import torch
import inspect

peft = pytest.importorskip('peft')
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from omni_drones.navigation.forest.qwen import (assistant_labels, lora_targets,
                                              visual_merger_targets, attach_training_lora)


@pytest.mark.parametrize('vision_adapter', [False, True])
def test_multimodal_lora_backward_and_adapter_reload(tmp_path, vision_adapter):
    config = Qwen3VLConfig(
        image_token_id=98, video_token_id=96, vision_start_token_id=99, vision_end_token_id=97,
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            rope_scaling={'rope_type': 'default', 'mrope_section': [3, 3, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0]))
    torch.manual_seed(42)
    base = Qwen3VLForConditionalGeneration(config)
    base.requires_grad_(False)
    model, targets = attach_training_lora(base, rank=2, vision_adapter=vision_adapter)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    # 一个图像占位 token 对应 2×2 原始 patch 合并后的视觉嵌入。
    ids = torch.tensor([[10, 99, 98, 97, 20, 21, 2]])
    labels = assistant_labels(ids, ids[:, :5])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
        pixel_values=torch.randn(4, 3*2*16*16), image_grid_thw=torch.tensor([[1, 2, 2]]))
    if 'mm_token_type_ids' in inspect.signature(base.forward).parameters:
        batch['mm_token_type_ids'] = (ids == config.image_token_id).long()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in trainable)
    visual = [(n,p) for n,p in model.named_parameters() if '.visual.' in n]
    assert all(not p.requires_grad and p.grad is None for n,p in visual if 'lora_' not in n)
    assert any(p.requires_grad and p.grad is not None and p.grad.count_nonzero()
               for n,p in visual) == vision_adapter
    optimizer.step()
    expected = peft.get_peft_model_state_dict(model)
    model.save_pretrained(tmp_path)
    restored = peft.PeftModel.from_pretrained(Qwen3VLForConditionalGeneration(config), tmp_path)
    actual = peft.get_peft_model_state_dict(restored)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name])


def test_language_warm_start_preserves_weights_and_initial_function_when_adding_merger(tmp_path):
    config = Qwen3VLConfig(
        image_token_id=98, video_token_id=96, vision_start_token_id=99, vision_end_token_id=97,
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            rope_scaling={'rope_type': 'default', 'mrope_section': [3, 3, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0]))
    torch.manual_seed(19)
    first = Qwen3VLForConditionalGeneration(config)
    base_weights = {name:value.clone() for name,value in first.state_dict().items()}
    source,_ = attach_training_lora(first, rank=2)
    with torch.no_grad():
        for name,param in source.named_parameters():
            if 'lora_B' in name:
                param.normal_(std=.02)
    source.save_pretrained(tmp_path)
    source.eval()
    second = Qwen3VLForConditionalGeneration(config)
    second.load_state_dict(base_weights)
    expanded,targets = attach_training_lora(second, rank=2, adapter=tmp_path, vision_adapter=True)
    expanded.eval()
    source_state = peft.get_peft_model_state_dict(source)
    expanded_state = peft.get_peft_model_state_dict(expanded)
    for name,value in source_state.items():
        torch.testing.assert_close(value, expanded_state[name], rtol=0, atol=0)
    assert all(not param.count_nonzero() for name,param in expanded.named_parameters()
               if '.visual.' in name and 'lora_B' in name)
    ids = torch.tensor([[10,99,98,97,20]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 pixel_values=torch.randn(4,3*2*16*16), image_grid_thw=torch.tensor([[1,2,2]]))
    if 'mm_token_type_ids' in inspect.signature(first.forward).parameters:
        batch['mm_token_type_ids'] = (ids == config.image_token_id).long()
    with torch.inference_mode():
        torch.testing.assert_close(source(**batch).logits, expanded(**batch).logits, rtol=0, atol=0)
    assert any('.visual.merger.' in name for name in targets)
