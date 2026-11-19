import math
import functools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPConfig, CLIPModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPAttention,
    CLIPEncoderLayer,
    CLIPMLP,
    CLIPTextEmbeddings,
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.utils import ModelOutput

from src.common import registry
from src.utils.utils import negative_contrastive_loss
from .configuration_delta import DELTAConfig

__all__ = [
    "DELTAOutput",
    "DELTAPreTrainedModel",
    "DELTAModel",
]


@dataclass
class DELTAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    cont_loss: Optional[torch.FloatTensor] = None
    text_order_loss: Optional[torch.FloatTensor] = None
    vision_binding_loss: Optional[torch.FloatTensor] = None
    fusion_gate_mean: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: Optional[BaseModelOutputWithPooling] = None
    vision_model_output: Optional[BaseModelOutputWithPooling] = None


@dataclass
class RelationAwareTextModelOutput(BaseModelOutputWithPooling):
    fusion_gate_mean: Optional[torch.FloatTensor] = None
    fusion_gate: Optional[torch.FloatTensor] = None


@dataclass
class RelationAwareEncoderOutput(BaseModelOutput):
    captured_hidden_state: Optional[torch.FloatTensor] = None


class DELTAPreTrainedModel(PreTrainedModel):
    config_class = DELTAConfig
    base_model_prefix = "delta"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable[..., torch.Tensor] = torch.utils.checkpoint.checkpoint,
    ) -> None:
        is_gradient_checkpointing_set = self._apply_gradient_checkpointing(
            self,
            enable=enable,
            gradient_checkpointing_func=gradient_checkpointing_func,
        )

        if not is_gradient_checkpointing_set:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    @staticmethod
    def _apply_gradient_checkpointing(
        root_module: nn.Module,
        enable: bool,
        gradient_checkpointing_func: Callable[..., torch.Tensor],
    ) -> bool:
        is_gradient_checkpointing_set = False

        for module in root_module.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable
                module._gradient_checkpointing_func = gradient_checkpointing_func
                is_gradient_checkpointing_set = True

            encoder = getattr(module, "encoder", None)
            if encoder is not None and hasattr(encoder, "gradient_checkpointing"):
                encoder.gradient_checkpointing = enable
                encoder._gradient_checkpointing_func = gradient_checkpointing_func
                is_gradient_checkpointing_set = True

        return is_gradient_checkpointing_set


class RelationAwareDELTAAttention(CLIPAttention):
    def __init__(self, config, text_binding_cfg: dict):
        super().__init__(config)
        self.num_buckets = int(text_binding_cfg.get("num_buckets", 32))
        self.max_distance = max(1, int(text_binding_cfg.get("max_distance", 77)))
        self.num_directions = 2

        self.gate_q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.gate_k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.logirpe_k = nn.Parameter(torch.zeros(self.num_directions, self.num_buckets, self.head_dim))
        self.logirpe_v = nn.Parameter(torch.zeros(self.num_directions, self.num_buckets, self.head_dim))
        self._relative_position_cache = {}

    def reset_relation_parameters(self):
        nn.init.zeros_(self.logirpe_k)
        nn.init.zeros_(self.logirpe_v)
        nn.init.xavier_uniform_(self.gate_q_proj.weight)
        nn.init.xavier_uniform_(self.gate_k_proj.weight)
        if self.gate_q_proj.bias is not None:
            nn.init.zeros_(self.gate_q_proj.bias)
        if self.gate_k_proj.bias is not None:
            nn.init.zeros_(self.gate_k_proj.bias)

    def _get_direction_index(
        self,
        tgt_len: int,
        src_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.training:
            context_position = torch.arange(tgt_len, device=device)[:, None]
            memory_position = torch.arange(src_len, device=device)[None, :]
            return (context_position > memory_position).to(torch.long).unsqueeze(0).unsqueeze(0)

        cache_key = (
            tgt_len,
            src_len,
            device.type,
            device.index if device.index is not None else -1,
        )
        direction_index = self._relative_position_cache.get(cache_key)
        if direction_index is None:
            context_position = torch.arange(tgt_len, device=device)[:, None]
            memory_position = torch.arange(src_len, device=device)[None, :]
            direction_index = (context_position > memory_position).to(torch.long)
            direction_index = direction_index.unsqueeze(0).unsqueeze(0)
            self._relative_position_cache[cache_key] = direction_index
        return direction_index

    def _get_token_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        bsz: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            src_valid = torch.ones(bsz, 1, 1, src_len, dtype=torch.bool, device=device)
        else:
            src_valid = attention_mask[:, :, :1, :].eq(0)

        if tgt_len == src_len:
            query_valid = src_valid.transpose(-1, -2)
        else:
            query_valid = torch.ones(bsz, 1, tgt_len, 1, dtype=torch.bool, device=device)
        return query_valid, src_valid

    def _build_bucket_positions(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()
        gate_query_states = self._shape(self.gate_q_proj(hidden_states), tgt_len, bsz).float()
        gate_key_states = self._shape(self.gate_k_proj(hidden_states), -1, bsz).float()
        src_len = gate_key_states.size(2)

        query_valid, src_valid = self._get_token_masks(
            attention_mask,
            bsz,
            tgt_len,
            src_len,
            hidden_states.device,
        )
        gate_logits = torch.matmul(gate_query_states, gate_key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        gates = torch.sigmoid(gate_logits)
        gates = gates * query_valid.to(dtype=gates.dtype) * src_valid.to(dtype=gates.dtype)

        prefix_counts = torch.cumsum(gates, dim=-1)
        anchor_counts = prefix_counts.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        contextual_distance = (anchor_counts - prefix_counts).abs()
        contextual_distance = contextual_distance * src_valid.to(dtype=contextual_distance.dtype)
        contextual_distance = contextual_distance.clamp(max=float(self.max_distance))
        bucket_positions = contextual_distance * (self.num_buckets - 1) / float(self.max_distance)
        return bucket_positions.to(dtype=hidden_states.dtype)

    def _flatten_bucket_indices(
        self,
        bucket_positions: torch.Tensor,
        direction_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_bucket = bucket_positions.floor().clamp(min=0, max=self.num_buckets - 1).to(torch.long)
        upper_bucket = torch.clamp(lower_bucket + 1, max=self.num_buckets - 1)
        alpha = bucket_positions - lower_bucket.to(dtype=bucket_positions.dtype)

        direction_index = direction_index.expand(
            bucket_positions.size(0),
            bucket_positions.size(1),
            bucket_positions.size(2),
            bucket_positions.size(3),
        )
        lower_flat = direction_index * self.num_buckets + lower_bucket
        upper_flat = direction_index * self.num_buckets + upper_bucket
        return lower_flat, upper_flat, alpha

    def _compute_key_bias_scores(
        self,
        query_states: torch.Tensor,
        lower_flat: torch.Tensor,
        upper_flat: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        q_beta = torch.einsum(
            "nhtd,urd->nhtur",
            query_states,
            self.logirpe_k.to(dtype=query_states.dtype, device=query_states.device),
        )
        q_beta = q_beta.reshape(
            query_states.size(0),
            query_states.size(1),
            query_states.size(2),
            self.num_directions * self.num_buckets,
        )

        lower_scores = q_beta.gather(-1, lower_flat)
        upper_scores = q_beta.gather(-1, upper_flat)
        return lower_scores + alpha * (upper_scores - lower_scores)

    def _compute_value_bias_output(
        self,
        attn_probs: torch.Tensor,
        lower_flat: torch.Tensor,
        upper_flat: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = attn_probs.new_zeros(
            attn_probs.size(0),
            attn_probs.size(1),
            attn_probs.size(2),
            self.num_directions * self.num_buckets,
        )
        coeffs.scatter_add_(-1, lower_flat, attn_probs * (1.0 - alpha))
        coeffs.scatter_add_(-1, upper_flat, attn_probs * alpha)
        coeffs = coeffs.view(
            attn_probs.size(0),
            attn_probs.size(1),
            attn_probs.size(2),
            self.num_directions,
            self.num_buckets,
        )
        return torch.einsum(
            "nhtur,urd->nhtd",
            coeffs,
            self.logirpe_v.to(dtype=attn_probs.dtype, device=attn_probs.device),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        src_len = key_states.size(2)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        bucket_positions = self._build_bucket_positions(hidden_states, attention_mask)
        direction_index = self._get_direction_index(tgt_len, src_len, hidden_states.device)
        lower_flat, upper_flat, alpha = self._flatten_bucket_indices(bucket_positions, direction_index)
        attn_weights = attn_weights + self._compute_key_bias_scores(
            query_states,
            lower_flat,
            upper_flat,
            alpha.to(dtype=query_states.dtype),
        )

        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights + causal_attention_mask

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights_reshaped = attn_weights if output_attentions else None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output + self._compute_value_bias_output(
            attn_probs,
            lower_flat,
            upper_flat,
            alpha.to(dtype=attn_probs.dtype),
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class RelationAwareDELTAEncoderLayer(nn.Module):
    def __init__(self, config, text_binding_cfg: dict):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = RelationAwareDELTAAttention(config, text_binding_cfg=text_binding_cfg)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class RelationAwareDELTAEncoder(nn.Module):
    def __init__(self, config, text_binding_cfg: dict):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        self.text_binding_enabled = bool(text_binding_cfg.get("enabled", False))
        self.fused_readout_enabled = bool(text_binding_cfg.get("fused_readout_enabled", True))
        layer_indices = text_binding_cfg.get("layer_indices", [4, 5, 6, 7])
        self.layer_indices = tuple(
            sorted(
                {
                    int(idx) for idx in layer_indices
                    if 0 <= int(idx) < config.num_hidden_layers
                }
            )
        ) if self.text_binding_enabled else ()
        if self.text_binding_enabled and not self.layer_indices:
            raise ValueError(
                "`text_binding.layer_indices` must contain at least one valid layer index."
            )
        self.readout_layer_index = self.layer_indices[-1] if self.layer_indices else None
        layers = []
        for idx in range(config.num_hidden_layers):
            if self.text_binding_enabled and idx in self.layer_indices:
                layers.append(RelationAwareDELTAEncoderLayer(config, text_binding_cfg=text_binding_cfg))
            else:
                layers.append(CLIPEncoderLayer(config))
        self.layers = nn.ModuleList(layers)

    def load_from_base(self, base_encoder) -> None:
        self.gradient_checkpointing = getattr(base_encoder, "gradient_checkpointing", False)
        self._gradient_checkpointing_func = getattr(
            base_encoder,
            "_gradient_checkpointing_func",
            torch.utils.checkpoint.checkpoint,
        )
        for layer, base_layer in zip(self.layers, base_encoder.layers):
            if isinstance(layer, RelationAwareDELTAEncoderLayer):
                layer.load_state_dict(base_layer.state_dict(), strict=False)
                layer.self_attn.reset_relation_parameters()
            else:
                layer.load_state_dict(base_layer.state_dict())

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        capture_hidden_state_layer: Optional[int] = None,
    ) -> Union[Tuple, RelationAwareEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        captured_hidden_state = None

        hidden_states = inputs_embeds
        for layer_index, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                checkpoint_fn = getattr(self, "_gradient_checkpointing_func", torch.utils.checkpoint.checkpoint)
                if (
                    checkpoint_fn is torch.utils.checkpoint.checkpoint
                    or getattr(checkpoint_fn, "func", None) is torch.utils.checkpoint.checkpoint
                ):
                    checkpoint_fn = functools.partial(checkpoint_fn, use_reentrant=False)

                layer_outputs = checkpoint_fn(
                    self._checkpointed_encoder_layer_forward(
                        encoder_layer=encoder_layer,
                        output_attentions=output_attentions,
                    ),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if capture_hidden_state_layer is not None and layer_index == capture_hidden_state_layer:
                captured_hidden_state = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return RelationAwareEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            captured_hidden_state=captured_hidden_state,
        )

    @staticmethod
    def _checkpointed_encoder_layer_forward(
        encoder_layer: nn.Module,
        output_attentions: bool,
    ) -> Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], Tuple[torch.Tensor, ...]]:
        def custom_forward(hs, am, cam):
            return encoder_layer(
                hs,
                am,
                cam,
                output_attentions=output_attentions,
            )

        return custom_forward


class MidTopTokenFusionReadout(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, gate_bias_init: float = -2.0):
        super().__init__()
        self.gate_bias_init = float(gate_bias_init)
        self.mid_norm = nn.LayerNorm(embed_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.readout = ResidualCrossAttentionReadout(
            embed_dim,
            num_heads=num_heads,
            normalize_output=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        first_linear = self.gate_mlp[0]
        final_linear = self.gate_mlp[-1]
        nn.init.xavier_uniform_(first_linear.weight)
        nn.init.zeros_(first_linear.bias)
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, self.gate_bias_init)

    def forward(
        self,
        mid_hidden_states: torch.Tensor,
        top_hidden_states: torch.Tensor,
        eos_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mid_hidden_states = self.mid_norm(mid_hidden_states)

        batch_indices = torch.arange(top_hidden_states.shape[0], device=top_hidden_states.device)
        mid_eos = mid_hidden_states[batch_indices, eos_indices]
        top_eos = top_hidden_states[batch_indices, eos_indices]

        gate_inputs = torch.cat([mid_eos, top_eos], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_inputs)).unsqueeze(-1)
        fused_hidden_states = top_hidden_states + gate * (mid_hidden_states - top_hidden_states)

        key_padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        pooled_output = self.readout(
            top_eos,
            fused_hidden_states,
            key_padding_mask=key_padding_mask,
        )
        return pooled_output, gate.squeeze(-1).squeeze(-1)


class RelationAwareTextTransformer(nn.Module):
    def __init__(self, config, text_binding_cfg: dict):
        super().__init__()
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = RelationAwareDELTAEncoder(config, text_binding_cfg)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.eos_token_id = config.eos_token_id
        self.text_binding_enabled = bool(text_binding_cfg.get("enabled", False))
        self.fused_readout_enabled = bool(text_binding_cfg.get("fused_readout_enabled", True))
        self.fused_readout = MidTopTokenFusionReadout(
            config.hidden_size,
            num_heads=config.num_attention_heads,
            gate_bias_init=text_binding_cfg.get("fused_readout_gate_bias_init", -2.0),
        )

    def load_from_base(self, base_text_model) -> None:
        self.embeddings.load_state_dict(base_text_model.embeddings.state_dict())
        self.encoder.load_from_base(base_text_model.encoder)
        self.final_layer_norm.load_state_dict(base_text_model.final_layer_norm.state_dict())
        self.eos_token_id = base_text_model.eos_token_id

    def _get_eos_indices(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.eos_token_id == 2:
            return input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)
        return (
            input_ids.to(dtype=torch.int, device=hidden_states.device) == self.eos_token_id
        ).int().argmax(dim=-1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RelationAwareTextModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        flat_attention_mask = None
        if attention_mask is not None:
            flat_attention_mask = attention_mask.view(-1, input_shape[-1])
            attention_mask = flat_attention_mask

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape,
            hidden_states.dtype,
            device=hidden_states.device,
        )

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        capture_hidden_state_layer = self.encoder.readout_layer_index if (
            self.text_binding_enabled and self.fused_readout_enabled
        ) else None
        encoder_output_hidden_states = bool(output_hidden_states)
        if not return_dict and capture_hidden_state_layer is not None:
            encoder_output_hidden_states = True
        should_capture_hidden_state = return_dict and not encoder_output_hidden_states

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=encoder_output_hidden_states,
            return_dict=return_dict,
            capture_hidden_state_layer=capture_hidden_state_layer if should_capture_hidden_state else None,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        eos_indices = self._get_eos_indices(input_ids, last_hidden_state)
        fusion_gate_mean = None
        fusion_gate = None
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            eos_indices,
        ]
        if self.text_binding_enabled and self.fused_readout_enabled:
            readout_layer_index = self.encoder.readout_layer_index
            if readout_layer_index is None:
                raise ValueError("`text_binding.layer_indices` must be set when fused readout is enabled.")
            if return_dict:
                if encoder_outputs.hidden_states is not None:
                    mid_hidden_state = encoder_outputs.hidden_states[readout_layer_index + 1]
                else:
                    mid_hidden_state = encoder_outputs.captured_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[1]
                mid_hidden_state = encoder_hidden_states[readout_layer_index + 1]
            if mid_hidden_state is None:
                raise ValueError("Failed to capture the fused readout hidden state.")
            pooled_output, fusion_gate = self.fused_readout(
                mid_hidden_states=mid_hidden_state,
                top_hidden_states=last_hidden_state,
                eos_indices=eos_indices,
                attention_mask=flat_attention_mask,
            )
            fusion_gate_mean = fusion_gate.mean()

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return RelationAwareTextModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            fusion_gate_mean=fusion_gate_mean,
            fusion_gate=fusion_gate,
        )


class ResidualCrossAttentionReadout(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, normalize_output: bool = True):
        super().__init__()
        while num_heads > 1 and embed_dim % num_heads != 0:
            num_heads //= 2
        self.normalize_output = normalize_output
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_states.dim() == 2:
            query_states = query_states.unsqueeze(1)
        attn_output, _ = self.attn(
            query_states,
            context_states,
            context_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = self.attn_norm(query_states + attn_output)
        ff_hidden = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ff_hidden).squeeze(1)
        if self.normalize_output:
            return F.normalize(hidden_states, dim=-1)
        return hidden_states


class VisionBindingHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        object_attention_temperature: float,
        attribute_attention_temperature: float,
        object_prior_gamma: float,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.object_query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attribute_query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.binding_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.object_query_norm = nn.LayerNorm(embed_dim)
        self.attribute_query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.object_attention_temperature = max(float(object_attention_temperature), eps)
        self.attribute_attention_temperature = max(float(attribute_attention_temperature), eps)
        self.object_prior_gamma = float(object_prior_gamma)
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in (
            self.object_query_proj,
            self.attribute_query_proj,
            self.key_proj,
            self.value_proj,
            self.binding_proj,
        ):
            nn.init.eye_(linear.weight)
        for layer_norm in (
            self.object_query_norm,
            self.attribute_query_norm,
            self.key_norm,
        ):
            layer_norm.weight.data.fill_(1.0)
            layer_norm.bias.data.zero_()

    @staticmethod
    def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, dim=-1)

    def _build_object_prior(
        self,
        object_queries: torch.Tensor,
        patch_keys: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.einsum("bd,bpd->bp", object_queries, patch_keys)
        logits = logits / self.object_attention_temperature
        return F.softmax(logits, dim=-1)

    def _read_attribute_visual(
        self,
        attribute_queries: torch.Tensor,
        patch_keys: torch.Tensor,
        patch_values: torch.Tensor,
        object_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum("bd,bpd->bp", attribute_queries, patch_keys)
        logits = logits / self.attribute_attention_temperature
        logits = logits + self.object_prior_gamma * torch.log(object_prior.clamp_min(self.eps))
        attention = F.softmax(logits, dim=-1)
        visual_embed = torch.einsum("bp,bpd->bd", attention, patch_values)
        return visual_embed, attention

    def forward(
        self,
        object_text_embeds: torch.Tensor,
        pos_attribute_text_embeds: torch.Tensor,
        neg_attribute_text_embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        object_queries = self._normalize_rows(self.object_query_norm(self.object_query_proj(object_text_embeds)))
        pos_attribute_queries = self._normalize_rows(
            self.attribute_query_norm(self.attribute_query_proj(pos_attribute_text_embeds))
        )
        neg_attribute_queries = self._normalize_rows(
            self.attribute_query_norm(self.attribute_query_proj(neg_attribute_text_embeds))
        )
        patch_keys = self._normalize_rows(self.key_norm(self.key_proj(patch_embeds)))
        patch_values = self.value_proj(patch_embeds)

        object_prior = self._build_object_prior(object_queries, patch_keys)
        pos_visual_embeds, _ = self._read_attribute_visual(
            pos_attribute_queries,
            patch_keys,
            patch_values,
            object_prior,
        )
        neg_visual_embeds, _ = self._read_attribute_visual(
            neg_attribute_queries,
            patch_keys,
            patch_values,
            object_prior,
        )

        pos_scores = torch.einsum("bd,bd->b", pos_attribute_queries, self.binding_proj(pos_visual_embeds))
        neg_scores = torch.einsum("bd,bd->b", neg_attribute_queries, self.binding_proj(neg_visual_embeds))
        return pos_scores, neg_scores


@registry.register_model("DELTAModel")
class DELTAModel(DELTAPreTrainedModel):
    config_class = DELTAConfig

    def __init__(self, config: DELTAConfig, load_pretrained_backbone: bool = True):
        super().__init__(config)

        backbone_config = CLIPConfig.from_dict(config.backbone_config)
        backbone_config.text_config._attn_implementation = "eager"
        if (
            getattr(backbone_config.vision_config, "_attn_implementation", "eager") == "eager"
            and hasattr(F, "scaled_dot_product_attention")
        ):
            backbone_config.vision_config._attn_implementation = "sdpa"

        backbone_pretrained_name = getattr(config, "backbone_pretrained_model_name_or_path", None)
        if load_pretrained_backbone and backbone_pretrained_name is not None:
            base_model = CLIPModel.from_pretrained(backbone_pretrained_name, config=backbone_config)
        else:
            base_model = CLIPModel(backbone_config)

        self.text_loss_enabled = bool(config.text_binding.get("loss_enabled", True))
        self.text_loss_detach_image = bool(config.text_binding.get("loss_detach_image", True))
        self.text_loss_weight = float(config.text_binding.get("loss_weight", 1.0))
        self.text_loss_temperature = float(config.text_binding.get("loss_temperature", 5.0))
        self.vision_binding_enabled = bool(config.vision_binding.get("enabled", False))
        self.vision_binding_loss_detach_text = bool(config.vision_binding.get("loss_detach_text", True))
        self.vision_binding_loss_weight = float(config.vision_binding.get("loss_weight", 1.0))
        self.vision_binding_loss_temperature = float(config.vision_binding.get("loss_temperature", 10.0))

        self.vision_model = base_model.vision_model
        self.visual_projection = base_model.visual_projection
        self.text_projection = base_model.text_projection
        self.logit_scale = base_model.logit_scale

        self.text_model = RelationAwareTextTransformer(backbone_config.text_config, config.text_binding)
        self.text_model.load_from_base(base_model.text_model)
        self.vision_binding_head = None
        self.vision_binding_text_projection = None
        self.vision_binding_visual_projection = None
        if self.vision_binding_enabled:
            self.vision_binding_text_projection = self._build_vision_binding_projection(
                input_dim=backbone_config.text_config.hidden_size,
                output_dim=self.text_projection.out_features,
                source_projection=self.text_projection,
            )
            self.vision_binding_visual_projection = self._build_vision_binding_projection(
                input_dim=backbone_config.vision_config.hidden_size,
                output_dim=self.visual_projection.out_features,
                source_projection=self.visual_projection,
            )
            self.vision_binding_head = VisionBindingHead(
                embed_dim=self.text_projection.out_features,
                object_attention_temperature=config.vision_binding.get("object_attention_temperature", 0.07),
                attribute_attention_temperature=config.vision_binding.get("attribute_attention_temperature", 0.07),
                object_prior_gamma=config.vision_binding.get("object_prior_gamma", 1.0),
            )

        super()._backward_compatibility_gradient_checkpointing()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.setdefault("load_pretrained_backbone", False)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable[..., torch.Tensor] = torch.utils.checkpoint.checkpoint,
    ) -> None:
        is_gradient_checkpointing_set = self._apply_gradient_checkpointing(
            self.text_model.encoder,
            enable=enable,
            gradient_checkpointing_func=gradient_checkpointing_func,
        )

        if not is_gradient_checkpointing_set:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    def _normalize_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        return embeds / torch.linalg.vector_norm(embeds, dim=-1, keepdim=True).clamp_min(1e-12)

    def _build_vision_binding_projection(
        self,
        input_dim: int,
        output_dim: int,
        source_projection: nn.Linear,
    ) -> nn.Linear:
        projection = nn.Linear(input_dim, output_dim, bias=False)
        projection.weight.data.copy_(source_projection.weight.data)
        return projection

    def _encode_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )

    def _encode_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> RelationAwareTextModelOutput:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def _get_image_embeds_from_outputs(self, vision_outputs: BaseModelOutputWithPooling) -> torch.Tensor:
        image_embeds = self.visual_projection(vision_outputs.pooler_output)
        return self._normalize_embeds(image_embeds)

    def _project_text_pooler(self, pooler_output: torch.Tensor) -> torch.Tensor:
        return self._normalize_embeds(self.text_projection(pooler_output))

    def _project_text_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._normalize_embeds(self.text_projection(hidden_states))

    def _project_patch_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._normalize_embeds(self.visual_projection(hidden_states))

    def _project_vision_binding_text_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.vision_binding_text_projection is None:
            return self._project_text_hidden_states(hidden_states)
        return self._normalize_embeds(self.vision_binding_text_projection(hidden_states))

    def _project_vision_binding_patch_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.vision_binding_visual_projection is None:
            return self._project_patch_hidden_states(hidden_states)
        return self._normalize_embeds(self.vision_binding_visual_projection(hidden_states))

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        mask = token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        numerator = (hidden_states * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1.0e-6)
        return numerator / denominator

    def _pad_text_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        target_seq_len: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        pad_width = target_seq_len - input_ids.size(1)
        if pad_width < 0:
            raise ValueError(f"Cannot pad from length {input_ids.size(1)} to smaller target length {target_seq_len}.")
        if pad_width == 0:
            return input_ids, attention_mask

        pad_token_id = getattr(self.text_model.config, "pad_token_id", 0)
        return (
            F.pad(input_ids, (0, pad_width), value=pad_token_id),
            F.pad(attention_mask, (0, pad_width), value=0),
        )

    def _encode_text_group(
        self,
        batch_specs: List[Tuple[str, torch.LongTensor, torch.Tensor]],
        requires_grad: bool,
    ) -> Dict[str, torch.Tensor]:
        if not batch_specs:
            return {}

        target_seq_len = max(input_ids.size(1) for _, input_ids, _ in batch_specs)
        merged_input_ids = []
        merged_attention_masks = []
        split_sizes: List[Tuple[str, int]] = []

        for name, input_ids, attention_mask in batch_specs:
            padded_input_ids, padded_attention_mask = self._pad_text_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_seq_len=target_seq_len,
            )
            merged_input_ids.append(padded_input_ids)
            merged_attention_masks.append(padded_attention_mask)
            split_sizes.append((name, input_ids.size(0)))

        merged_input_ids = torch.cat(merged_input_ids, dim=0)
        merged_attention_masks = torch.cat(merged_attention_masks, dim=0)

        def run_encoder() -> torch.Tensor:
            outputs = self._encode_text(
                input_ids=merged_input_ids,
                attention_mask=merged_attention_masks,
                return_dict=True,
            )
            return outputs.pooler_output

        if requires_grad:
            merged_pooler_outputs = run_encoder()
        else:
            with torch.no_grad():
                merged_pooler_outputs = run_encoder()

        output: Dict[str, torch.Tensor] = {}
        start_idx = 0
        for name, batch_size in split_sizes:
            end_idx = start_idx + batch_size
            output[name] = merged_pooler_outputs[start_idx:end_idx]
            start_idx = end_idx
        return output

    def _encode_auxiliary_text_embeds(
        self,
        text_swap_input_ids: Optional[torch.LongTensor] = None,
        text_swap_attention_mask: Optional[torch.Tensor] = None,
        text_swap_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if (
            self.text_loss_enabled
            and text_swap_input_ids is not None
            and text_swap_attention_mask is not None
            and text_swap_valid_mask is not None
            and torch.any(text_swap_valid_mask > 0)
        ):
            pooled_batches = self._encode_text_group(
                [("text_swap", text_swap_input_ids, text_swap_attention_mask)],
                requires_grad=True,
            )
            return {
                name: self._project_text_pooler(pooler_output)
                for name, pooler_output in pooled_batches.items()
            }
        return {}

    def _compute_vision_binding_loss(
        self,
        text_hidden_states: torch.Tensor,
        patch_hidden_states: torch.Tensor,
        positive_count: int,
        vision_binding_valid_mask: Optional[torch.Tensor] = None,
        vision_binding_neg_text_indices: Optional[torch.LongTensor] = None,
        vision_binding_object_token_mask: Optional[torch.Tensor] = None,
        vision_binding_pos_attribute_token_mask: Optional[torch.Tensor] = None,
        vision_binding_neg_attribute_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = patch_hidden_states.new_zeros(())
        if not (
            self.vision_binding_enabled
            and self.vision_binding_head is not None
            and vision_binding_valid_mask is not None
            and vision_binding_neg_text_indices is not None
            and vision_binding_object_token_mask is not None
            and vision_binding_pos_attribute_token_mask is not None
            and vision_binding_neg_attribute_token_mask is not None
        ):
            return loss

        valid_mask = vision_binding_valid_mask.to(device=patch_hidden_states.device, dtype=patch_hidden_states.dtype)
        if not torch.any(valid_mask > 0):
            return loss

        neg_text_indices = vision_binding_neg_text_indices.to(device=text_hidden_states.device, dtype=torch.long)
        if neg_text_indices.numel() != positive_count:
            raise ValueError("`vision_binding_neg_text_indices` must match the positive batch size.")

        positive_text_hidden_states = text_hidden_states[:positive_count]
        negative_text_hidden_states = text_hidden_states.index_select(0, neg_text_indices)
        if self.vision_binding_loss_detach_text:
            positive_text_hidden_states = positive_text_hidden_states.detach()
            negative_text_hidden_states = negative_text_hidden_states.detach()

        object_token_mask = vision_binding_object_token_mask.to(device=text_hidden_states.device)
        pos_attribute_token_mask = vision_binding_pos_attribute_token_mask.to(device=text_hidden_states.device)
        neg_attribute_token_mask = vision_binding_neg_attribute_token_mask.to(device=text_hidden_states.device)

        object_text_embeds = self._project_vision_binding_text_hidden_states(
            self._masked_mean_pool(positive_text_hidden_states, object_token_mask)
        )
        pos_attribute_text_embeds = self._project_vision_binding_text_hidden_states(
            self._masked_mean_pool(positive_text_hidden_states, pos_attribute_token_mask)
        )
        neg_attribute_text_embeds = self._project_vision_binding_text_hidden_states(
            self._masked_mean_pool(negative_text_hidden_states, neg_attribute_token_mask)
        )
        patch_embeds = self._project_vision_binding_patch_hidden_states(patch_hidden_states)

        pos_scores, neg_scores = self.vision_binding_head(
            object_text_embeds=object_text_embeds,
            pos_attribute_text_embeds=pos_attribute_text_embeds,
            neg_attribute_text_embeds=neg_attribute_text_embeds,
            patch_embeds=patch_embeds,
        )
        score_margin = pos_scores - neg_scores
        per_example = -F.logsigmoid(self.vision_binding_loss_temperature * score_margin)
        loss = (per_example * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return loss

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = True if return_dict is None else return_dict
        text_outputs = self._encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooler_output = text_outputs.pooler_output if return_dict else text_outputs[1]
        return self._project_text_pooler(pooler_output)

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        vision_outputs = self._encode_image(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        return self._get_image_embeds_from_outputs(vision_outputs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_swap_input_ids: Optional[torch.LongTensor] = None,
        text_swap_attention_mask: Optional[torch.Tensor] = None,
        text_swap_valid_mask: Optional[torch.Tensor] = None,
        vision_binding_valid_mask: Optional[torch.Tensor] = None,
        vision_binding_neg_text_indices: Optional[torch.LongTensor] = None,
        vision_binding_object_token_mask: Optional[torch.Tensor] = None,
        vision_binding_pos_attribute_token_mask: Optional[torch.Tensor] = None,
        vision_binding_neg_attribute_token_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ) -> Union[Tuple, DELTAOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self._encode_image(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        text_outputs = self._encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        image_embeds = self._get_image_embeds_from_outputs(vision_outputs)
        text_embeds = self._project_text_pooler(text_outputs.pooler_output)
        positive_count = image_embeds.size(0)
        auxiliary_text_embeds = self._encode_auxiliary_text_embeds(
            text_swap_input_ids=text_swap_input_ids,
            text_swap_attention_mask=text_swap_attention_mask,
            text_swap_valid_mask=text_swap_valid_mask,
        )

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(
            text_embeds.device
        )
        logits_per_image = logits_per_text.t()
        cont_loss = negative_contrastive_loss(logits_per_image)

        text_order_loss = torch.zeros((), device=image_embeds.device, dtype=image_embeds.dtype)
        if (
            self.text_loss_enabled
            and text_swap_input_ids is not None
            and text_swap_attention_mask is not None
            and text_swap_valid_mask is not None
        ):
            valid_mask = text_swap_valid_mask.to(device=image_embeds.device, dtype=image_embeds.dtype)
            if torch.any(valid_mask > 0):
                text_swap_embeds = auxiliary_text_embeds.get("text_swap")
                if text_swap_embeds is None:
                    text_swap_outputs = self._encode_text(
                        input_ids=text_swap_input_ids,
                        attention_mask=text_swap_attention_mask,
                        return_dict=True,
                    )
                    text_swap_embeds = self._project_text_pooler(text_swap_outputs.pooler_output)
                image_anchor = image_embeds.detach() if self.text_loss_detach_image else image_embeds
                s_pos = (image_anchor * text_embeds[:positive_count]).sum(dim=-1)
                s_neg = (image_anchor * text_swap_embeds).sum(dim=-1)
                per_example = -F.logsigmoid(self.text_loss_temperature * (s_pos - s_neg))
                text_order_loss = (per_example * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

        vision_binding_loss = self._compute_vision_binding_loss(
            text_hidden_states=text_outputs.last_hidden_state,
            patch_hidden_states=vision_outputs.last_hidden_state[:, 1:, :],
            positive_count=positive_count,
            vision_binding_valid_mask=vision_binding_valid_mask,
            vision_binding_neg_text_indices=vision_binding_neg_text_indices,
            vision_binding_object_token_mask=vision_binding_object_token_mask,
            vision_binding_pos_attribute_token_mask=vision_binding_pos_attribute_token_mask,
            vision_binding_neg_attribute_token_mask=vision_binding_neg_attribute_token_mask,
        )

        total_loss = (
            cont_loss
            + self.text_loss_weight * text_order_loss
            + self.vision_binding_loss_weight * vision_binding_loss
        )

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (total_loss, cont_loss, text_order_loss, vision_binding_loss) + output

        return DELTAOutput(
            loss=total_loss,
            cont_loss=cont_loss,
            text_order_loss=text_order_loss,
            vision_binding_loss=vision_binding_loss,
            fusion_gate_mean=getattr(text_outputs, "fusion_gate_mean", None),
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
