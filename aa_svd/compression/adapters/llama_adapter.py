# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import copy
import gc
import logging
from math import cos, pi
from typing import List

import torch
import tqdm
import wandb
from torch import FloatTensor
from torch.nn import Linear, Module
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
)

from aa_svd.compression.compressed_linear import CompressedLinear
from aa_svd.compression.model_adapter import LayerAdapter, ModelAdapter, HeadLayerAdapterMixin
from aa_svd.compression.utils import map_tensors

logger = logging.getLogger(__name__)


class LlamaLayerAdapter(LayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer, modules_to_replace: List[str]) -> None:
        super().__init__(modules_to_replace=modules_to_replace)
        self._layer: LlamaDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def _build_compression_groups(self) -> dict[Module, List[Module]]:
        return {
            self.layer.self_attn.q_proj: [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj],
            self.layer.self_attn.o_proj: [self.layer.self_attn.o_proj],
            self.layer.mlp.gate_proj:    [self.layer.mlp.gate_proj, self.layer.mlp.up_proj],
            self.layer.mlp.down_proj:    [self.layer.mlp.down_proj],
        }


class LlamaHeadAdapter(HeadLayerAdapterMixin, LlamaLayerAdapter):
    def __init__(self, layer: Module, modules_to_replace: List[str]) -> None:
        super().__init__(layer=layer, modules_to_replace=modules_to_replace)

    def _build_compression_groups(self) -> dict[Module, List[Module]]:
        return {self.layer.lm_head: [self.layer.lm_head]}


class LlamaModelAdapter(ModelAdapter):
    def __init__(self, model: LlamaForCausalLM, modules_to_replace: List[str]) -> None:
        super().__init__(modules_to_replace=modules_to_replace)
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def layer_adapter_type(self) -> type:
        return LlamaLayerAdapter

    @property
    def use_cache(self) -> bool:
        return self._model.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self._model.config.use_cache = value

    def get_layers(self) -> list[LayerAdapter]:
        adapters = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Filter the configured module names to only those that are present in this layer
            if self._modules_to_replace is None:
                layer_modules_to_replace = None
            else:
                layer_modules_to_replace = [m for m in self._modules_to_replace if m.startswith(f"model.layers.{layer_idx}.")]

            adapters.append(self.layer_adapter_type(layer, modules_to_replace=layer_modules_to_replace))
        return adapters

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> List[Module]:
        return [self.model.model.embed_tokens]

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Llama-2 and Llama-3 don't have a pad token by default
        tokenizer.pad_token = tokenizer.eos_token
        self._model.config.pad_token_id = tokenizer.pad_token_id

    def get_layer_name_prefix(self, idx) -> str:
        """
        Returns the prefix used to identify layers in the model. For example, in Llama, this would be "model.layers."
        """
        return f"model.layers.{idx}."

    def get_last_layer_to_output_adapter(self) -> Module:
        """
        Returns the module that produces the final output from the last
        layer's output. For Llama this wraps the model's final
        normalization (RMSNorm) and the language-model head (lm_head)
        so compression can target the lm_head.
        """
        layer = self.get_last_layer_to_output_wrapper()

        # Candidate exposed names on the wrapper. The wrapper exposes the
        # final model normalization as `norm` and the linear head as
        # `head` (backed by the model's `lm_head`). When the user supplies
        # configured module names, accept either exact matches or names
        # that end with the candidate (e.g. "model.norm" or "lm_head").
        candidate_modules = ["norm", "lm_head"]

        if self._modules_to_replace is None:
            layer_modules_to_replace = None
        else:
            def _matches_candidate(name: str) -> bool:
                return any(name == c or name.endswith(f".{c}") for c in candidate_modules)

            layer_modules_to_replace = [m for m in self._modules_to_replace if _matches_candidate(m)]

        adapter = LlamaHeadAdapter(layer, modules_to_replace=layer_modules_to_replace)
        return adapter

    def get_last_layer_to_output_wrapper(self) -> Module:
        """
        Returns a wrapper module that takes the last layer's output and
        produces the final logits. The wrapper exposes `norm` (the
        final RMSNorm on the inner model) and `head` (the top-level
        `lm_head`) so adapters can discover and replace them.
        """

        class _HeadWrapper(Module):
            """Wraps Llama's final normalization and lm_head so the adapter
            can treat them like a single layer. The wrapper exposes two
            registered submodules: `norm` (the inner model's RMSNorm) and
            `head` (the top-level `lm_head` Linear), when present.
            """

            def __init__(self, _model: Module, norm: Module = None, lm_head: Module = None) -> None:
                super().__init__()
                # Keep a reference to the full original model without
                # registering it as a submodule to avoid duplicating
                # entries in named_modules().
                object.__setattr__(self, "_model", _model)

                # Expose the final layer norm from the inner model if
                if norm is None:
                    self.norm = getattr(_model.model, "norm")
                else:
                    self.norm = norm

                # Expose the language-model head (lm_head) that produces
                # logits from hidden states. This is typically a Linear.
                if lm_head is None:
                    self.lm_head = getattr(_model, "lm_head", None)
                else:
                    self.lm_head = lm_head

            def custom_clone(self) -> '_HeadWrapper':
                return _HeadWrapper(
                    _model=self._model,
                    norm=copy.deepcopy(self.norm),
                    lm_head=copy.deepcopy(self.lm_head)
                )

            def forward(self, *args, **kwargs) -> FloatTensor:
                """Apply final norm (if present) and return a
                BaseModelOutputWithPast containing the normalized hidden
                states. This mirrors LlamaForCausalLM where the
                `lm_head` is applied outside the inner model using
                `outputs.last_hidden_state`.

                Expects hidden_states of shape (batch, seq_len, hidden_dim).
                Returns a BaseModelOutputWithPast with `last_hidden_state`
                set to the normalized hidden states and `past_key_values`
                forwarded from kwargs when present.
                """
                x = args[0]
                x = self.norm(x)

                # Forward any provided past_key_values (if present) so the
                # caller can access them from the returned object.
                past_key_values = kwargs.get("past_key_values", None)

                output = BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values)
                hidden_states = output.last_hidden_state

                logits_to_keep = 0
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

                return logits

        return _HeadWrapper(self._model)

    def finetune_layer(self, layer_adapter, layer_args, layer_kwargs, targets, device: torch.device | str,
                      layer_idx, num_steps: int = 100, lr: float = 1e-4, loss_type: str = 'ce'):
        layer_adapter.layer.to(device)

        optimizer = torch.optim.AdamW(layer_adapter.layer.parameters(), lr=lr, weight_decay=0.01)

        # Define schedule
        total_steps = num_steps*len(layer_args)
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))

        # Create scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # loss_fn = torch.nn.CrossEntropyLoss()

        pbar = tqdm.tqdm(range(num_steps), total=num_steps, desc="Fine-tuning layer adapter")
        for step in pbar:
            epoch_loss = 0.
            for i, (layer_args_batch, layer_kwargs_batch, target_logits) in enumerate(zip(layer_args, layer_kwargs, targets)):
                layer_args_batch, layer_kwargs_batch, target_logits = map_tensors(
                                                            [layer_args_batch, layer_kwargs_batch, target_logits], device=device
                                                        )
                logits = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
                # hidden_state_pos = layer_adapter.hidden_states_output_position
                # logits = out_batch[hidden_state_pos]

                # print('logits shape:', logits.shape)
                # print('target_logits shape:', target_logits.shape)

                if loss_type == 'mse':
                    loss = torch.nn.functional.mse_loss(logits, target_logits)
                    pbar.set_postfix({'mse_loss': loss.item()})
                elif loss_type == 'kl':
                    log_probs = torch.nn.functional.log_softmax(logits / 1.0, dim=-1)
                    target_probs = torch.nn.functional.softmax(target_logits / 1.0, dim=-1)
                    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='batchmean') * (1.0 ** 2)
                    pbar.set_postfix({'kl_loss': loss.item()})
                elif loss_type == 'ce':
                    target_probits = torch.softmax(target_logits, dim=-1)
                    target_probits = target_probits.view(-1, target_probits.size(-1))
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_probits)
                    pbar.set_postfix({'ce_loss': loss.item(), 'ppl': torch.exp(loss).item()})
                elif loss_type == 'cosine':
                    logits_flat = logits.view(-1, logits.size(-1))
                    target_logits_flat = target_logits.view(-1, target_logits.size(-1))
                    loss = 1 - torch.nn.functional.cosine_similarity(logits_flat, target_logits_flat, dim=-1).mean()
                    pbar.set_postfix({'cosine_loss': loss.item()})
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

                epoch_loss += loss.item()

                if wandb.run is not None:
                    wandb.log({
                        f"finetune_step_loss/layer_{layer_idx}": loss.item(),
                        "finetune_step": step*len(layer_args) + i
                    })

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(layer_adapter.layer.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # del batch, original_outputs, adapted_outputs, loss
                torch.cuda.empty_cache()

            if wandb.run is not None:
                wandb.log({
                    f"finetune_epoch_loss/layer_{layer_idx}": epoch_loss,
                    "finetune_epoch": step})

            pbar.set_postfix({'epoch_loss': epoch_loss})

        layer_adapter.layer.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    def finetune_layer_with_allocation(
        self,
        layer_adapter,
        layer_args,
        layer_kwargs,
        targets,
        device: torch.device | str,
        layer_idx: int,
        target_ratio: float,
        num_steps: int = 100,
        lr: float = 1e-3,
        lr_k: float = 5.0,
        loss_type: str = 'ce',
        lambda_ratio: float = 200.0,
        beta: float = 10.0,
    ):
        """Fine-tune a layer while jointly learning per-layer rank allocation.

        Follows Algorithm 1 of Dobi-SVD (Wang et al., ICLR 2025).  For each
        CompressedLinear in *layer_adapter* a scalar learnable truncation
        boundary ``k`` is introduced.  During the forward pass the i-th rank
        component (0-indexed, ordered by descending singular value) is weighted
        by the smooth step::

            mask_i = 0.5 * tanh(beta * (k - i)) + 0.5

        so the effective output is::

            out = U( V(x) * mask )

        ``k`` is optimised jointly with the layer parameters.  A ratio penalty::

            lambda_ratio * (current_ratio - target_ratio)^2

        with ``current_ratio = k * (out + in) / (out * in)`` drives ``k``
        toward the target.

        After optimisation ``k`` is rounded to the nearest integer and the first
        ``k`` columns of U (resp. rows of V) are kept — no gate absorption is
        needed because the SVD components are already ordered by importance.

        Args:
            target_ratio: Desired compression ratio k(m+n)/(mn) (0 < ratio ≤ 1).
            lambda_ratio: Penalty weight for the ratio constraint.
            beta: Sharpness of the tanh smooth truncation (paper default: 10).
        """
        import torch.nn as nn

        layer_adapter.layer.to(device)

        # ------------------------------------------------------------------
        # Collect CompressedLinear modules
        # ------------------------------------------------------------------
        compressed_linears: dict[str, CompressedLinear] = {
            name: module
            for name, module in layer_adapter.layer.named_modules()
            if isinstance(module, CompressedLinear)
        }

        if not compressed_linears:
            logger.warning(
                "finetune_layer_with_allocation: no CompressedLinear modules found "
                "— falling back to finetune_layer."
            )
            self.finetune_layer(
                layer_adapter, layer_args, layer_kwargs, targets,
                device, layer_idx, num_steps, lr, loss_type,
            )
            return

        # ------------------------------------------------------------------
        # Create scalar truncation-boundary parameters k (one per layer).
        # Initialise at the target rank implied by target_ratio:
        #   k = target_ratio * m*n / (m+n)  (inverse of r = k(m+n)/(m*n))
        #
        # Always float32: with beta=10 the tanh gradient at the boundary is
        # ~sech²(10) ≈ 8e-9, which underflows to exactly zero in fp16/bf16.
        # ------------------------------------------------------------------
        k_params: dict[str, torch.nn.Parameter] = {
            name: torch.nn.Parameter(
                torch.tensor(
                    float(torch.randint(1, min(module.out_features, module.in_features) + 1, (1,)).item()),
                    device=device,
                    dtype=torch.float32,
                )
            )
            for name, module in compressed_linears.items()
        }

        # ------------------------------------------------------------------
        # Patch forwards: apply Dobi-SVD smooth truncation mask between V and U.
        # The mask is computed in float32 to keep gradients alive — with beta=10
        # the sech² values can be ~1e-8, which underflows in bf16/fp16.
        # ------------------------------------------------------------------
        original_forwards: dict[str, object] = {}

        def _make_tanh_forward(mod: CompressedLinear, k_param: torch.nn.Parameter):
            r = mod.rank
            # indices are fixed; precompute in fp32
            indices_fp32 = torch.arange(r, device=device, dtype=torch.float32)

            def forward(x: torch.Tensor) -> torch.Tensor:
                # clamp k to [1, r] — stays in fp32, gradient flows through clamp
                k_clamped = k_param.clamp(1.0, float(r))
                # compute mask in fp32 to avoid underflow in the tanh backward
                mask_fp32 = 0.5 * torch.tanh(beta * (k_clamped - indices_fp32)) + 0.5 # (r,)
                h = mod.V(x)                        # (..., r) in x.dtype
                h = h * mask_fp32.to(x.dtype)       # broadcast; cast mask after grad computation
                return mod.U(h)
            return forward

        for name, module in compressed_linears.items():
            original_forwards[name] = module.forward
            module.forward = _make_tanh_forward(module, k_params[name])

        # ------------------------------------------------------------------
        # Optimiser + scheduler  (layer params + k params)
        # Two param groups: k lives on the scale of the rank (O(100s)), so it
        # needs lr_k >> lr_weights.  No weight decay on k (decay → 0 is wrong).
        # ------------------------------------------------------------------
        all_params = (
            list(layer_adapter.layer.parameters())
            + list(k_params.values())
        )
        optimizer = torch.optim.AdamW([
            {'params': list(layer_adapter.layer.parameters()), 'lr': lr, 'weight_decay': 0.01},
            {'params': list(k_params.values()),                'lr': lr_k, 'weight_decay': 0.0},
        ])

        total_steps = num_steps * len(layer_args)
        warmup_steps = int(0.1 * total_steps)

        # lr schedule for UV weights: warmup then cosine decay
        def lr_lambda_weights(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))

        # lr schedule for k: constant — k needs to travel O(rank) units so
        # warmup from 0 wastes budget, and decay would prevent it from settling
        def lr_lambda_k(step):
            return 1.0

        # one lambda per param group (same order as optimizer param groups)
        scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda_weights, lr_lambda_k])

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        global_step = 0
        pbar = tqdm.tqdm(range(num_steps), total=num_steps, desc="Fine-tuning layer w/ rank allocation")
        for step in pbar:
            epoch_task_loss = 0.0
            for i, (layer_args_batch, layer_kwargs_batch, target_logits) in enumerate(
                zip(layer_args, layer_kwargs, targets)
            ):
                layer_args_batch, layer_kwargs_batch, target_logits = map_tensors(
                    [layer_args_batch, layer_kwargs_batch, target_logits], device=device
                )

                logits = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)

                # ---- task loss ----
                if loss_type == 'mse':
                    task_loss = torch.nn.functional.mse_loss(logits, target_logits)
                elif loss_type == 'kl':
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    target_probs = torch.nn.functional.softmax(target_logits, dim=-1)
                    task_loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='batchmean')
                elif loss_type == 'ce':
                    target_probits = torch.softmax(target_logits, dim=-1).view(-1, target_logits.size(-1))
                    task_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), target_probits
                    )
                elif loss_type == 'cosine':
                    task_loss = 1 - torch.nn.functional.cosine_similarity(
                        logits.view(-1, logits.size(-1)),
                        target_logits.view(-1, target_logits.size(-1)),
                        dim=-1,
                    ).mean()
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

                # ---- ratio penalty (global across all compressed layers in this transformer layer) ----
                # current_ratio = sum_l k_l*(m_l+n_l) / sum_l m_l*n_l
                # (Dobi-SVD eq: r = k(m+n)/(m*n))
                current_params = sum(
                    (k_params[name].clamp(1.0, float(module.rank)))
                    * (module.out_features + module.in_features)
                    for name, module in compressed_linears.items()
                )
                total_params = sum(
                    module.out_features * module.in_features
                    for module in compressed_linears.values()
                )
                current_ratio = current_params / total_params
                ratio_loss = lambda_ratio * torch.abs(current_ratio - target_ratio)

                loss = task_loss + ratio_loss
                global_step += 1
                epoch_task_loss += task_loss.item()

                optimizer.zero_grad()
                loss.backward()

                # ---- gradient diagnostics ----
                k_grad_norm = torch.norm(
                    torch.stack([kp.grad.float() if kp.grad is not None else torch.zeros(1, device=device)
                                 for kp in k_params.values()])
                )
                total_grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

                # # per-module U/V grad norms — verify layer weights are live
                # uv_grad_norms = {}
                # for n, mod in compressed_linears.items():
                #     u_gn = mod.U.weight.grad.float().norm().item() if mod.U.weight.grad is not None else float('nan')
                #     v_gn = mod.V.weight.grad.float().norm().item() if mod.V.weight.grad is not None else float('nan')
                #     uv_grad_norms[n] = (u_gn, v_gn)

                # uv_str = " | ".join(f"{n}: U={u:.2e} V={v:.2e}" for n, (u, v) in uv_grad_norms.items())
                # logger.info(f"[step {step}] UV grads — {uv_str}")
                # logger.info(f"[step {step}] k_gnorm={k_grad_norm.item():.2e}  tot_gnorm={total_grad_norm.item():.2e}")

                pbar.set_postfix({
                    f'{loss_type}': f'{task_loss.item():.3e}',
                    'ratio_loss': f'{ratio_loss.item():.3e}',
                    'cr': f'{current_ratio.item():.3f}',
                    'k_gnorm': f'{k_grad_norm.item():.2e}',
                    'tot_gnorm': f'{total_grad_norm.item():.2e}',
                })

                if wandb.run is not None:
                    wandb.log({
                        f"finetune_step_loss/layer_{layer_idx}": task_loss.item(),
                        f"finetune_ratio_loss/layer_{layer_idx}": ratio_loss.item(),
                        f"finetune_alloc_current_ratio/layer_{layer_idx}": current_ratio.item(),
                        "finetune_step": step * len(layer_args) + i,
                        f"finetune_alloc_k_grad_norm/layer_{layer_idx}": k_grad_norm.item(),
                        f"finetune_alloc_total_grad_norm/layer_{layer_idx}": total_grad_norm.item(),
                        **{f"finetune_alloc_k/layer_{layer_idx}/{n}": k_params[n].item() for n in k_params},
                    })

                optimizer.step()
                scheduler.step()
                torch.cuda.empty_cache()

            if wandb.run is not None:
                wandb.log({
                    f"finetune_epoch_loss/layer_{layer_idx}": epoch_task_loss,
                    "finetune_epoch": step,
                })

        # ------------------------------------------------------------------
        # Restore original forwards before trimming
        # ------------------------------------------------------------------
        for name, module in compressed_linears.items():
            module.forward = original_forwards[name]

        # ------------------------------------------------------------------
        # Trim: slice U / V to the first round(k) components.
        #
        # SVD components are stored in descending singular-value order, so the
        # first k columns of U.weight and rows of V.weight are the most
        # important — no gate absorption is needed.
        # ------------------------------------------------------------------
        for name, module in compressed_linears.items():
            r = module.rank
            k_raw = k_params[name].detach().clamp(1.0, float(r)).item()
            k = max(1, int(k_raw))

            #   U.weight : (out_features, r)  ->  (out_features, k)
            #   V.weight : (r, in_features)   ->  (k, in_features)
            new_U = nn.Linear(k, module.out_features, bias=(module.U.bias is not None), dtype=module.U.weight.dtype)
            new_V = nn.Linear(module.in_features, k, bias=False, dtype=module.V.weight.dtype)

            new_U.weight = nn.Parameter(module.U.weight.data[:, :k].clone())
            new_V.weight = nn.Parameter(module.V.weight.data[:k, :].clone())

            if module.U.bias is not None:
                new_U.bias = nn.Parameter(module.U.bias.data.clone())

            module.U = new_U.to(module.U.weight.device)
            module.V = new_V.to(module.V.weight.device)
            module.rank = k

            actual_ratio = k * (module.out_features + module.in_features) / (module.out_features * module.in_features)
            logger.info(
                f"[layer {layer_idx}] {name}: rank {r} → {k} "
                f"(k_learned={k_raw:.2f}, layer param ratio={actual_ratio:.3f})"
            )

            if wandb.run is not None:
                wandb.log({
                    f"finetune_alloc_final_k/{name}": k,
                    f"finetune_alloc_final_ratio/{name}": actual_ratio,
                    "layer_idx": layer_idx
                })

        layer_adapter.layer.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
