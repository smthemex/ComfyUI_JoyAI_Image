from dataclasses import dataclass, field
from pathlib import Path

from .src.infer_runtime.infer_config import InferConfig


# def _resolve_root() -> Path:
#     here = Path(__file__).resolve().parent
#     if (here / "transformer").exists() and (here / "vae").exists() and (here / "JoyAI-Image-Und").exists():
#         return here
#     raise ValueError(
#         "Place this config file directly inside the checkpoint root."
#     )


# _ROOT = _resolve_root()


@dataclass
class JoyAIImageInferConfig(InferConfig):
    dit_arch_config: dict = field(
        default_factory=lambda: {
            "target": "modules.models.Transformer3DModel",
            "params": {
                "hidden_size": 4096,
                "in_channels": 16,
                "heads_num": 32,
                "mm_double_blocks_depth": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "rope_dim_list": [16, 56, 56],
                "text_states_dim": 4096,
                "rope_type": "rope",
                "dit_modulation_type": "wanx",
                "theta": 10000,
                "attn_backend": "flash_attn",
            },
        }
    )
    # vae_arch_config: dict = field(
    #     default_factory=lambda: {
    #         "target": "modules.models.WanxVAE",
    #         "params": {
    #             "pretrained": str(_ROOT / "vae" / "Wan2.1_VAE.pth"),
    #         },
    #     }
    # )
    # text_encoder_arch_config: dict = field(
    #     default_factory=lambda: {
    #         "target": "modules.models.load_text_encoder",
    #         "params": {
    #             "text_encoder_ckpt": str(_ROOT / "JoyAI-Image-Und"),
    #         },
    #     }
    # )
    scheduler_arch_config: dict = field(
        default_factory=lambda: {
            "target": "modules.models.FlowMatchDiscreteScheduler",
            "params": {
                "num_train_timesteps": 1000,
                "shift": 4.0,
            },
        }
    )

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 2048

    # Keep these fields visible in the active config because they control multi-GPU inference.
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False
    use_fsdp_inference: bool = False
    cpu_offload: bool = False
    pin_cpu_memory: bool = False
