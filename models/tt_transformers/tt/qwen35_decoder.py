# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Decoder block for Gated DeltaNet (linear attention) layers in Qwen3.5.

Qwen3.5 uses a hybrid architecture: 3/4 DeltaNet layers + 1/4 full attention layers.
Full attention layers use the standard TransformerBlock with GatedAttention.
DeltaNet layers use this DeltaNetDecoderBlock, which has the same forward signature
as TransformerBlock but routes to GatedDeltaNet instead of standard attention.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup


class DeltaNetDecoderBlock(LightweightModule):
    """Decoder block wrapping GatedDeltaNet + MLP for Qwen3.5 linear attention layers.

    Has the same forward signature as TransformerBlock so the model forward loop
    doesn't need per-layer dispatch logic.
    """

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        prefetcher=None,
        attention_class=None,
        mlp_class=None,
        mlp_dtype=None,
        mlp_weight_cache_path=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.prefetcher = prefetcher
        self.layer_num = layer_num

        # Attention: DeltaNet (default) or GatedAttention (for full_attention layers)
        if attention_class is not None:
            self.attention = attention_class(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
                transformation_mats=None,
                configuration=args,
            )
        else:
            self.attention = GatedDeltaNet(
                mesh_device=mesh_device,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
            )
            self.attention.initialize_states()

        # MLP: use custom class (e.g. Qwen35MoE) or standard MLP
        mlp_cls = mlp_class or MLP
        self.feed_forward = mlp_cls(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=mlp_weight_cache_path or weight_cache_path,
            layer_num=layer_num,
            dtype=mlp_dtype or dtype,
            model_config=args.get_model_config(),
            prefetcher=prefetcher,
        )

        # Layer norms (same as TransformerBlock)
        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="ffn_norm",
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        # DeltaNet layers ignore: current_pos, rot_mats, page_table, kv_cache
        #
        # Memory config: use the same DRAM-sharded residual config as the standard
        # TransformerBlock. The norm and MLP are configured to work with this layout.
        _d(f"[D{self.layer_num}:")
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        x = ttnn.to_memory_config(x, skip_mem_cfg)
        residual = x

        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = self.attention_norm(x, mode, norm_config=attn_norm_config)

        if hasattr(self.attention, "initialize_states"):
            attn_out = self.attention.forward(attn_in)
        else:
            attn_out = self.attention.forward(attn_in, current_pos=current_pos, rot_mats=rot_mats_global, mode=mode)

        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        hidden_states = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
        residual = hidden_states
        if mode == Mode.PREFILL:
            x.deallocate(True)
        ttnn.deallocate(attn_out)

        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)
        hidden_states = self.feed_forward.forward(hidden_states, mode)

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=activation_dtype or ttnn.bfloat16,
        )
        return out
