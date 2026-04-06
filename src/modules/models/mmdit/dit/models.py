from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
import gc 
import copy
from ....models.attention import attention, get_cu_seqlens

from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .modulate_layers import load_modulation, modulate, apply_gate



class BlockGPUManager:
    def __init__(self, device="cuda", block_group_size=1):

        self.device = torch.device(device)
        self.managed_modules = [] 
        self.submodule = []    
        self.block_group_size = block_group_size  # 每次加载的连续层数
        self._original_model_ref = None
        self._original_block_ref = None
        self._num_groups = 0  # 总批次数
        self._group_loaded: list[bool] = [] 
        

    def setup_for_inference(self, transformer_model):
        self._collect_managed_modules(transformer_model)
        self._initialize_submodule()
        return self
    

    def _collect_managed_modules(self, transformer_model):
        self.submodule = []
        self._original_model_ref = transformer_model
        self._original_block_ref = transformer_model.double_blocks

        self._num_groups = (len(self._original_block_ref) + self.block_group_size - 1) // self.block_group_size
    
        for attr in ['img_in', 'condition_embedder', 'norm_out', 'proj_out', ]:
            if hasattr(transformer_model, attr):
                self.submodule.append(getattr(transformer_model, attr))

        self.managed_modules = [None] * self._num_groups
        self._group_loaded = [False] * self._num_groups

    def _load_group(self, group_index):
        """加载指定组的数据块"""
        if self._group_loaded[group_index]:
            return
        
        start_idx = group_index * self.block_group_size
        end_idx = min(start_idx + self.block_group_size, len(self._original_block_ref))
        
        group = nn.ModuleList()
        for layer in self._original_block_ref[start_idx:end_idx]:
            # 深拷贝当前层
            cpu_layer = copy.deepcopy(layer)
            # 移动到目标设备
            cpu_layer.to(self.device)
            group.append(cpu_layer)
        
        self.managed_modules[group_index] = group
        self._group_loaded[group_index] = True

    def _unload_group(self, group_index):
        """卸载指定组的数据块"""
        if not self._group_loaded[group_index]:
            return
        
        group = self.managed_modules[group_index]
        self.managed_modules[group_index] = None
        self._group_loaded[group_index] = False
        
        # 显式删除引用
        group = None
        del group
        
        # 清理GPU缓存
        torch.cuda.empty_cache()


    def _get_layer(self, layer_index):
        """按需获取层，实现按组加载"""
        group_index = layer_index // self.block_group_size
        local_idx = layer_index % self.block_group_size
        
        # 如果组未加载，则加载该组
        if not self._group_loaded[group_index]:
            self._load_group(group_index)
        
        # 返回组中的指定层
        return self.managed_modules[group_index][local_idx]

    def _unload_unused_groups(self, keep: set[int]):
        """卸载不需要的组"""
        for index in range(self._num_groups):
            if self._group_loaded[index] and index not in keep:
                self._unload_group(index)
                
    def _initialize_submodule(self):
        for module in self.submodule:
            if hasattr(module, 'to'):
                module.to(self.device)
        return self
    
    def unload_all_blocks_to_cpu(self):
        # 卸载所有组
        for group_index in range(self._num_groups):
            self._unload_group(group_index)
        
        # 将embedder和output模块移到CPU
        for module in self.submodule:
            if hasattr(module, 'to'):
                module.to('cpu', non_blocking=True)
        
        torch.cuda.empty_cache()
        return self



class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: Optional[str] = "wanx",
        attn_backend: str = 'flash_attn',
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_backend = attn_backend
        self.dit_modulation_type = dit_modulation_type
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = load_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size,
            factor=6,
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        # There is no dtype fpr FeedForward, because FSDP2 casts the dtype for all parameters.
        # You may need to give the dtype when no autocast and fsdp !!!
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn="gelu-approximate")

        self.txt_mod = load_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size,
            factor=6,
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True,
                                       eps=1e-6, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn="gelu-approximate")

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple = None,
        txt_freqs_cis: tuple = None,
        attn_kwargs: Optional[dict] = {},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tt, th, tw = attn_kwargs['thw']
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if vis_freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(
                img_q, img_k, vis_freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        if txt_freqs_cis is not None:
            raise NotImplementedError("RoPE text is not supported for inference")
            txt_qq, txt_kk = apply_rotary_emb(
                txt_q, txt_k, txt_freqs_cis, head_first=False)
            assert (
                txt_qq.shape == txt_q.shape and txt_kk.shape == txt_k.shape
            ), f"txt_kk: {txt_qq.shape}, txt_q: {txt_q.shape}, txt_kk: {txt_kk.shape}, txt_k: {txt_k.shape}"
            txt_q, txt_k = txt_qq, txt_kk

        # attention computation start

        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        attn = attention(
            q, k, v,
            backend=self.attn_backend,
            attn_kwargs=attn_kwargs,
        )
        attn = attn.flatten(2, 3)
        # attention computation end
        img_attn, txt_attn = attn[:,
                                    : img.shape[1]], attn[:, img.shape[1]:]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn),
                               gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn),
                               gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin):
    _fsdp_shard_conditions: list = [
        lambda name, module: isinstance(module, (MMDoubleStreamBlock))]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 20,
        rope_dim_list: List[int] = [16, 56, 56],
        rope_type: str = 'rope',
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: str = "wanx",
        attn_backend: str = 'flash_attn',
        theta: int = 256,
    ):
        self.args = args
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = rope_dim_list
        self.dit_modulation_type = dit_modulation_type
        self.mm_double_blocks_depth = mm_double_blocks_depth
        self.attn_backend = attn_backend
        self.rope_type = rope_type
        self.theta = theta

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )

        # image projection
        self.img_in = nn.Conv3d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # condition embedding
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_states_dim,
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    dit_modulation_type=self.dit_modulation_type,
                    attn_backend=attn_backend,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # Output norm & projection
        self.norm_out = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            hidden_size, out_channels * math.prod(patch_size),
            **factory_kwargs)


    def get_rotary_pos_embed(self, vis_rope_size, txt_rope_size=None):
        target_ndim = 3
        ndim = 5 - 2

        if len(vis_rope_size) != target_ndim:
            vis_rope_size = [1] * (target_ndim - len(vis_rope_size)
                                   ) + vis_rope_size  # time axis
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim //
                             target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        vis_freqs, txt_freqs = get_nd_rotary_pos_embed(
            rope_dim_list,
            vis_rope_size,
            txt_rope_size=txt_rope_size,
            theta=self.theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return vis_freqs, txt_freqs

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,  # Should be in range(0, 1000).
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        return_dict: bool = True,
        gpu_manager: Optional[BlockGPUManager] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # For Multi-item Input: hidden_states: (b, n, c, t, h, w)
        # Permute the items into the temporal dimension
        is_multi_item = (len(hidden_states.shape) == 6)
        num_items = 0
        if is_multi_item:
            num_items = hidden_states.shape[1]
            if num_items > 1:
                assert self.patch_size[0] == 1, "For multi-item input, patch_size[0] must be 1"
                # Move the last item to the first position
                hidden_states = torch.cat(
                    [
                        hidden_states[:, -1:],
                        hidden_states[:, :-1]
                    ],
                    dim=1
                )
            hidden_states = rearrange(
                hidden_states, 'b n c t h w -> b c (n t) h w')

        out = {}
        batch_size, _, ot, oh, ow = hidden_states.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        # Text Mask
        if encoder_hidden_states_mask == None:
            encoder_hidden_states_mask = torch.ones(
                (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]), dtype=torch.bool).to(encoder_hidden_states.device)

        # Prepare img, txt, vec.
        img = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        temb, vec, txt = self.condition_embedder(
            timestep, encoder_hidden_states)
        if vec.shape[-1] > self.hidden_size:
            vec = vec.unflatten(1, (6, -1))

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # rope
        vis_freqs_cis, txt_freqs_cis = self.get_rotary_pos_embed(vis_rope_size=(
            tt, th, tw), txt_rope_size=txt_seq_len if self.rope_type == 'mrope' else None)

        # Compute attn_kwargs
        attn_kwargs = {'thw': [tt, th, tw], 'txt_len': txt_seq_len}
        if self.attn_backend == 'flash_attn':
            cu_seqlens_q = get_cu_seqlens(
                encoder_hidden_states_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = img_seq_len + txt_seq_len
            max_seqlen_kv = max_seqlen_q

            attn_kwargs.update({
                'cu_seqlens_q': cu_seqlens_q,
                'cu_seqlens_kv': cu_seqlens_kv,
                'max_seqlen_q': max_seqlen_q,
                'max_seqlen_kv': max_seqlen_kv,
            })

        # --------------------- Pass through DiT blocks ------------------------
        #for _, block in enumerate(self.double_blocks):
        for layer_index in range(len(self.double_blocks)):
            if gpu_manager is not None:
                block = gpu_manager._get_layer(layer_index)
            else:
                block = self.double_blocks[layer_index]

            double_block_args = [
                img,
                txt,
                vec,
                vis_freqs_cis,
                txt_freqs_cis,
                attn_kwargs
            ]

            img, txt = block(*double_block_args)
            if gpu_manager is not None:
                current_group = layer_index // gpu_manager.block_group_size
                next_group = current_group + 1
                gpu_manager._unload_unused_groups(keep={current_group, next_group})

        img_len = img.shape[1]
        x = torch.cat((img, txt), 1)
        img = x[:, :img_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.proj_out(self.norm_out(img))

        img = self.unpatchify(img, tt, th, tw)


        # Reshape back to multiple items
        if is_multi_item:
            img = rearrange(
                img, 'b c (n t) h w -> b n c t h w', n=num_items)
            if num_items > 1:
                # Move the first item back to the last position
                img = torch.cat(
                    [
                        img[:, 1:],
                        img[:, :1]
                    ],
                    dim=1
                )

        return (img, txt)

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = torch.einsum("nthwopqc->nctohpwq", x)

        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs
