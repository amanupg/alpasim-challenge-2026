# Teacher-forced flow-matching forward for Alpamayo 1.5.
# Adapted from NVlabs/alpamayo-recipes, recipes/alpamayo1_5_quant/quantize.py
# (_am1_5_ext_teacher_forced_flow_loss_forward), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License, Version 2.0. Changes: standalone function; the VLM prefill can run without
# gradients (frozen VLM), so only the action expert and its projections are trained.
from typing import Any

import einops
import torch
from alpamayo1_5.models.token_utils import to_special_token


def teacher_forced_flow(model, data: dict[str, Any], vlm_grad: bool = False) -> dict[str, torch.Tensor]:
    """v_pred / v_target (B, n_diffusion_tokens, action_dim) for one batch; MSE between them is the loss.

    data: tokenized_data (input_ids + processor outputs), ego_history_xyz/rot (B,1,16,...), ego_future_xyz/rot
    (B,1,64,...), all in the t0 rig frame, as Alpamayo's own loader builds them."""
    ego_history_xyz, ego_history_rot = data["ego_history_xyz"], data["ego_history_rot"]
    ego_future_xyz, ego_future_rot = data["ego_future_xyz"], data["ego_future_rot"]
    B, n_traj_group, _, _ = ego_history_xyz.shape
    assert n_traj_group == 1, "Only one trajectory group is supported."

    tokenized_data = dict(data["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot})
    device = input_ids.device
    start_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], 1), start_id, dtype=input_ids.dtype, device=device)], 1)
    if tokenized_data.get("attention_mask") is not None:
        am = tokenized_data["attention_mask"]
        tokenized_data["attention_mask"] = torch.cat([am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)], 1)

    with torch.set_grad_enabled(vlm_grad):
        vlm_outputs = model.vlm(input_ids=input_ids, use_cache=True, return_dict=True, **tokenized_data)
    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    rope_deltas = model.vlm.model.rope_deltas

    n_tok = model.action_space.get_action_space_dims()[0]
    offset = torch.full((B,), prefill_seq_len, device=device, dtype=torch.long)
    position_ids = einops.repeat(torch.arange(n_tok, device=device), "l -> 3 b l", b=B).clone()
    position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)
    attention_mask = torch.zeros((B, 1, n_tok, prefill_seq_len + n_tok), dtype=torch.float32, device=device)
    fwd = {"is_causal": False} if model.config.expert_non_causal_attention else {}

    x_1 = model.action_space.traj_to_action(traj_history_xyz=ego_history_xyz[:, 0], traj_history_rot=ego_history_rot[:, 0],
                                            traj_future_xyz=ego_future_xyz[:, 0], traj_future_rot=ego_future_rot[:, 0])
    x_1 = x_1.to(device=device, dtype=torch.float32)
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, 1, 1, device=device, dtype=x_1.dtype)
    x_t = (1.0 - t) * x_0 + t * x_1
    v_target = x_1 - x_0

    pd = next(model.action_in_proj.parameters()).dtype
    emb = model.action_in_proj(x_t.to(pd), t.to(pd))
    if emb.dim() == 2:
        emb = emb.view(B, n_tok, -1)
    out = model.expert(inputs_embeds=emb, position_ids=position_ids, past_key_values=prompt_cache,
                       attention_mask=attention_mask, use_cache=True, **fwd)
    prompt_cache.crop(prefill_seq_len)
    v_pred = model.action_out_proj(out.last_hidden_state[:, -n_tok:]).view(B, *model.action_space.get_action_space_dims())
    return {"v_pred": v_pred.to(torch.float32), "v_target": v_target}
