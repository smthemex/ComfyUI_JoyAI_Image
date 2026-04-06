"""Local inference entrypoint for the image understanding capability of JoyAI-Image."""

from __future__ import annotations

import argparse
from .src.modules.utils import _dynamic_resize_from_bucket
import time
import warnings
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor,Qwen3VLConfig
import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from transformers import  AutoTokenizer
import os
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
from safetensors.torch import load_file
import gc   
cur_path = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = Path(__file__).resolve().parent
# SRC_DIR = ROOT_DIR / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

from PIL import Image

warnings.filterwarnings("ignore")




def load_images(image_arg: str) -> list[Image.Image]:
    paths = [p.strip() for p in image_arg.split(",")]
    images = []
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        images.append(Image.open(p).convert("RGB"))
    return images


def resolve_text_encoder_path(ckpt_root: str) -> Path:
    root = Path(ckpt_root).expanduser().resolve()
    text_encoder_dir = root / "JoyAI-Image-Und"
    if not text_encoder_dir.is_dir():
        raise FileNotFoundError(
            f"Expected text_encoder/ directory inside checkpoint root: {root}"
        )
    return text_encoder_dir


def build_conversation(
    images: list[Image.Image],
    prompt: str | None,
) -> list[dict]:
    SYS_PROMPT = "You are a helpful assistant."
    
    default_prompt = "Describe this image in detail."
    user_text = prompt if prompt is not None else default_prompt

    image_content = [{"type": "image", "image": img} for img in images]
    user_content = image_content + [{"type": "text", "text": user_text}]

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages

joy_prompt_template_encode = {
    'image': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    'multiple_images': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n{}<|im_start|>assistant\n",
    'video': "<|im_start|>system\n \\nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
}
joy_prompt_template_encode_start_idx = {
            'image': 34,
            'multiple_images': 34,
            'video': 91,
        }


def extract_masked_hidden( hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

def get_qwen_prompt_embeds(text_encoder,tokenizer,
        prompt: Union[str, List[str]] = None,
        template_type: str = 'image',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):

    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = joy_prompt_template_encode[template_type]
    drop_idx = joy_prompt_template_encode_start_idx[template_type]
    txt = [template.format(e) for e in prompt]
    txt_tokens = tokenizer(
        txt, max_length=2048 + drop_idx, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    encoder_hidden_states = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    split_hidden_states = extract_masked_hidden(
        hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(
        e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = min([
        2048,
        max([u.size(0) for u in split_hidden_states]),
        max([u.size(0) for u in attn_mask_list])])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, encoder_attention_mask



def encode_prompt_multiple_images(
        text_encoder,qwen_processor,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        images: Optional[torch.Tensor] = None,
        template_type: Optional[str] = 'multiple_images',
        max_sequence_length: Optional[int] = None,
        drop_vit_feature: Optional[float] = False,
    ):
    assert template_type == 'multiple_images', "template_type must be 'multiple_images'"
    device = device 
    template = joy_prompt_template_encode[template_type]
    drop_idx = joy_prompt_template_encode_start_idx[template_type]
    prompt = [p.replace('<image>\n', '<|vision_start|><|image_pad|><|vision_end|>') for p in prompt]
    prompt = [template.format(p) for p in prompt]

    inputs = qwen_processor(
        text=prompt,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)
    encoder_hidden_states = text_encoder(
        **inputs,
        output_hidden_states=True,
    )
    last_hidden_states = encoder_hidden_states.hidden_states[-1]
    if drop_vit_feature:
        input_ids = inputs['input_ids']
        vlm_image_end_idx = torch.where(input_ids[0] == 151653)[0][-1]
        drop_idx = vlm_image_end_idx + 1
    prompt_embeds = last_hidden_states[:, drop_idx:]
    prompt_embeds_mask = inputs['attention_mask'][:, drop_idx:]
    if max_sequence_length is not None and prompt_embeds.shape[1] > max_sequence_length:
        prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
        prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]
    return prompt_embeds, prompt_embeds_mask

def encode_prompt(
    text_encoder, 
    prompt: Union[str, List[str]],
    images: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 4096,
    template_type: str = 'image',
    drop_vit_feature: bool = False,
    ):
    r"""

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_videos_per_prompt (`int`):
            number of videos that should be generated per prompt
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
    """
    qwen_processor = AutoProcessor.from_pretrained(os.path.join(cur_path, 'JoyAI-Image-Und'))
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(cur_path, 'JoyAI-Image-Und'),
        local_files_only=True,
        trust_remote_code=True,
    )
    if images is not None:
        ##################################################
        # from PIL import Image
        # images = [img.resize((512, 512), Image.LANCZOS) for img in images]
        ##################################################
        return encode_prompt_multiple_images(text_encoder,qwen_processor,
            prompt=prompt,
            images=images,
            device=device,
            max_sequence_length=max_sequence_length,
            drop_vit_feature=drop_vit_feature,
        )


    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(
        prompt) if prompt_embeds is None else prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask = get_qwen_prompt_embeds(text_encoder,tokenizer,
            prompt, template_type, device)

    prompt_embeds = prompt_embeds[:, :max_sequence_length]
    prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_videos_per_prompt, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.repeat(
        1, num_videos_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(
        batch_size * num_videos_per_prompt, seq_len)

    return prompt_embeds, prompt_embeds_mask




def load_qwen3vl_model(safetensors_path,gguf_path ) -> torch.nn.Module:
    #text_encoder_path = resolve_text_encoder_path(text_encoder_path)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    
    device = torch.device('cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading MLLM from: {safetensors_path}")
    print(f"Device: {device}, dtype: {dtype}")
    configs=Qwen3VLConfig.from_pretrained(os.path.join(cur_path, 'JoyAI-Image-Und'),local_files_only=True,model_type=dtype,trust_remote_code=True,low_cpu_mem_usage=False)
    with ctx():
        model = Qwen3VLForConditionalGeneration(configs)
    if safetensors_path is not None :
        model_dict=load_file(safetensors_path)
        match_state_dict(model, model_dict,show_num=20)
        x,y=model.load_state_dict(model_dict,strict=False,assign=True)
        print(x,"########_missing \n")
        print(y,"########_unused \n")
        del model_dict
        gc.collect()

        model.eval().to(dtype)
    elif gguf_path is not None:
        g_dict=load_gguf_checkpoint(gguf_path)
        match_state_dict(model, g_dict,show_num=20)
        set_gguf2meta_model(model,g_dict,dtype,torch.device("cpu"))
        del g_dict
        gc.collect()
    else:
        raise ValueError(
            "Please provide either a safetensors_path or a gguf_path."
        )
    return model

def encoder_input(model,prompt,images,max_new_tokens,top_p,top_k,temperature,infer_device):
    device = torch.device(infer_device) 
    processor = AutoProcessor.from_pretrained(
        os.path.join(cur_path, 'JoyAI-Image-Und'),
        local_files_only=True,
        trust_remote_code=True,
    )

    #images = load_images(args.image)
    messages = build_conversation(images, prompt)

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text_input],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print("Generating...")

    start_time = time.time()

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
    )
    if temperature == 0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        generate_kwargs["top_k"] = top_k

    if infer_device=="cuda":
        model.to("cuda")
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    if infer_device=="cuda":
        model.to("cpu")

    # Strip the input tokens from the output
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    elapsed = time.time() - start_time
    num_output_tokens = generated_ids.shape[1]

    print(f"\n{'=' * 60}")
    print(f"Response:\n{response}")
    print(f"{'=' * 60}")
    print(f"Output tokens: {num_output_tokens}")
    print(f"Time: {elapsed:.2f}s ({num_output_tokens / elapsed:.1f} tok/s)")

    # if save_response:
    #     output_path = Path(output)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     output_path.write_text(response, encoding="utf-8")
    #     print(f"Saved response to: {output_path}")
    return response

def get_conditioning(clip,prompt, images,infer_device):
    device = torch.device(infer_device) 
    num_items = 1 if images is None or len(
        images) == 0 else 1 + len(images)
    default_negative_prompt = ""
    if images is None:
        prompts = [f"<|im_start|>user\n{prompt}<|im_end|>\n"]
        negative_prompt = [f"<|im_start|>user\n{default_negative_prompt}<|im_end|>\n"]
    else:
        images = _dynamic_resize_from_bucket(images[0], basesize=1024)
        images.save("temp_image.png")
        width, height = images.size

        image_tokens = '<image>\n'
        prompts = [f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n"]
        negative_prompt = [f"<|im_start|>user\n{image_tokens}{default_negative_prompt}<|im_end|>\n"]
        # if num_items <= 1:
        #     negative_prompt = [
        #         f"<|im_start|>user\n{default_negative_prompt}<|im_end|>\n"] * 1
        # else:
        #     image_tokens = "<image>\n" * (num_items - 1)
        #     negative_prompt = [
        #         f"<|im_start|>user\n{image_tokens}{default_negative_prompt}<|im_end|>\n"] * 1
    if infer_device=="cuda":
        clip.to("cuda")
    
    prompt_embeds, prompt_embeds_mask=encode_prompt(clip,prompts,images,device)
    #print(prompt_embeds.shape, prompt_embeds_mask.shape) #torch.Size([1, 1037, 4096]) torch.Size([1, 1037])
    n_prompt_embeds, n_prompt_embeds_mask=encode_prompt(clip,negative_prompt,images,device)
    if infer_device=="cuda":
        clip.to("cpu")
    torch.cuda.empty_cache()
    positive=[[prompt_embeds,{"prompt_attention_mask": prompt_embeds_mask}]]
    negative=[[n_prompt_embeds,{"prompt_attention_mask": n_prompt_embeds_mask}]]
    return positive,negative

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


