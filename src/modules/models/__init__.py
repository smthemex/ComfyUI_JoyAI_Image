import os
import glob
import torch
#import torch.distributed as dist
import gc
from .bucket import BucketGroup
from .mmdit.dit import Transformer3DModel
from .mmdit.text_encoder import load_text_encoder
from .mmdit.vae import WanxVAE
from .pipeline import Pipeline
from .scheduler import FlowMatchDiscreteScheduler
from ..utils.fsdp_load import maybe_load_fsdp_model, pt_weights_iterator, safetensors_weights_iterator
from ..utils.logging import get_logger
from ..utils.constants import PRECISION_TO_TYPE
from ..utils.utils import build_from_config
from transformers import  AutoTokenizer
from safetensors.torch import load_file

def load_pipeline(cfg, dit,repo,):
    # vae
    #factory_kwargs = {
    #     'torch_dtype': PRECISION_TO_TYPE[cfg.vae_precision], "device": device}
    # vae = build_from_config(cfg.vae_arch_config, **factory_kwargs)
    # if getattr(cfg.vae_arch_config, "enable_feature_caching", False):
    #     vae.enable_feature_caching()

    # text_encoder
    # factory_kwargs = {
    #     'torch_dtype': PRECISION_TO_TYPE[cfg.text_encoder_precision], "device": device}
    # tokenizer, text_encoder = build_from_config(
    #     cfg.text_encoder_arch_config, **factory_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        local_files_only=True,
        trust_remote_code=True,
    )

    # scheduler
    #scheduler = build_from_config(cfg.scheduler_arch_config)
    scheduler = FlowMatchDiscreteScheduler(**cfg.scheduler_arch_config["params"])
    cfg.repo = repo

    pipeline = Pipeline(
        vae=None,
        tokenizer=tokenizer,
        text_encoder=None,
        transformer=dit,
        scheduler=scheduler,
        args=cfg,
    )

    #pipeline = pipeline.to(device)
    return pipeline


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    """Load DiT model with FSDP support."""
    logger = get_logger()
    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    model_kwargs = {'dtype': dtype, 'device': device, 'args': cfg}
    #model = build_from_config(cfg.dit_arch_config, **model_kwargs)
    with torch.device('meta'):
        model=Transformer3DModel(**model_kwargs,**cfg.dit_arch_config["params"])
    state_dict = None
    use_gguf=False
    if cfg.dit_ckpt is not None:
        logger.info(f"Loading model from: {cfg.dit_ckpt}")

        if cfg.dit_ckpt.endswith(".safetensors"):
            # Find all safetensors files
            # safetensors_files = glob.glob(
            #     os.path.join(str(cfg.dit_ckpt), "*.safetensors"))
            # if not safetensors_files:
            #     raise ValueError(
            #         f"No safetensors files found in {cfg.dit_ckpt}")
            # state_dict = dict(
            #     safetensors_weights_iterator(cfg.dit_ckpt))
            state_dict=load_file(cfg.dit_ckpt)
        elif cfg.dit_ckpt.endswith(".pth"):
            # pt_files = [cfg.dit_ckpt]
            # state_dict = dict(pt_weights_iterator(pt_files))
            state_dict=torch.load(cfg.dit_ckpt, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
        elif cfg.dit_ckpt.endswith(".gguf"):
            state_dict=load_gguf_checkpoint(cfg.dit_ckpt)
            use_gguf=True
        else:
            raise ValueError(
                f"Unknown checkpoint format: {cfg.dit_ckpt}, must be 'safetensor' or 'pth' or 'gguf'")

    # if not dist.is_initialized() or dist.get_world_size() == 1:
    #     # Debug mode
    #     model.to(device=device)
    match_state_dict(model,state_dict)

    if state_dict is not None:
        # filter unused params
        if not use_gguf:
            load_state_dict = {}
            for k, v in state_dict.items():
                if k == "img_in.weight" and model.img_in.weight.shape != v.shape:
                    logger.info(
                        f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                    v_new = v.new_zeros(model.img_in.weight.shape)
                    v_new[:, :v.shape[1], :, :, :] = v
                    v = v_new
                load_state_dict[k] = v
            model.load_state_dict(load_state_dict, strict=False,assign=True)
        else:
            model = set_gguf2meta_model(model,state_dict,dtype,device)

    # model = maybe_load_fsdp_model(
    #     model=model,
    #     hsdp_shard_dim=cfg.hsdp_shard_dim,
    #     reshard_after_forward=cfg.reshard_after_forward,
    #     param_dtype=dtype,
    #     reduce_dtype=torch.float32,
    #     output_dtype=None,
    #     cpu_offload=cfg.cpu_offload,
    #     fsdp_inference=cfg.use_fsdp_inference,
    #     training_mode=cfg.training_mode,
    #     pin_cpu_memory=cfg.pin_cpu_memory,
    # )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    # Ensure consistent dtype
    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        logger.warning(
            f"Model has mixed dtypes: {param_dtypes}. Converting to {dtype}")
        model = model.to(dtype)

    return model.eval()

def load_gguf_checkpoint(gguf_checkpoint_path):

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
  
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        quant_type = tensor.tensor_type

        
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data) #tensor.data.copy()
 
        parsed_parameters[name.replace("model.", "")] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
        del tensor,weights
        if i > 0 and i % 1000 == 0:  # 每1000个tensor执行一次gc
            logger.info(f"Processed {i}tensors...")
            gc.collect()
    del reader
    gc.collect()
    return parsed_parameters

def set_gguf2meta_model(meta_model,model_state_dict,dtype,device):
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True


    hf_quantizer._process_model_before_weight_loading(
        meta_model,
        device_map={"": device} if device else None,
        state_dict=model_state_dict
    )
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    x,y=load_model_dict_into_meta(
        meta_model, 
        model_state_dict, 
        hf_quantizer=hf_quantizer,
        device_map={"": device} if device else None,
        dtype=dtype
    )
    print(x,"offload_index")
    print(y,"state_dict_index")

    hf_quantizer._process_model_after_weight_loading(meta_model)

    
    del model_state_dict
    gc.collect()
    return meta_model.to(dtype=dtype)

def match_state_dict(meta_model, sd,show_num=10):

    meta_model_keys = set(meta_model.state_dict().keys())   
    state_dict_keys = set(sd.keys())

    # 打印匹配的键的数量
    matching_keys = meta_model_keys.intersection(state_dict_keys)
    print(f"Matching keys count: {len(matching_keys)}")
    
    # 打印不在 meta_model 中但在 state_dict 中的键（多余键）
    extra_keys = state_dict_keys - meta_model_keys
    if extra_keys:
        print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
        for key in list(extra_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 打印不在 state_dict 中但在 meta_model 中的键（缺失键）
    missing_keys = meta_model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
        for key in list(missing_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 如果需要，也可以打印部分匹配的键
    print(f"Sample matching keys: {list(matching_keys)[:5]}")


__all__ = [
    "BucketGroup",
    "FlowMatchDiscreteScheduler",
    "Pipeline",
    "Transformer3DModel",
    "WanxVAE",
    "load_pipeline",
    "load_text_encoder",
]
