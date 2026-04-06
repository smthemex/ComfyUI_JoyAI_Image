"""Local inference entrypoint for the clean JoyAI-Image release."""

from __future__ import annotations
import os
import time
import numpy as np
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from .src.infer_runtime.model import InferenceParams, build_model
from .src.infer_runtime.settings import InferSettings
from .src.modules.models.mmdit.vae import WanxVAE
from .model_loader_utils import map_0_1_to_neg1_1,map_neg1_1_to_0_1
cur_path = os.path.dirname(os.path.abspath(__file__))

def load_vae(vae_path,device,dtype ):
    vae=WanxVAE(vae_path,dtype,device)
    return vae

joy_ai_mean = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ]

joy_ai_std=[
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ]

def vae_decode(vae,latents):
    latents=latents["samples"] if isinstance(latents, dict) else latents
    if isinstance(vae, WanxVAE):
        with torch.autocast(device_type="cuda", dtype=vae.dtype, enabled=True):
            image = vae.decode(latents, return_dict=False)[0]
            #print(f" {image.shape}, dtype: {image.dtype}, device: {image.device}")
            image = rearrange(image, "(b n) c f h w -> b n c f h w", b=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float().permute(0, 1, 3, 2, 4, 5)
        # image_tensor = (image[0, -1, 0] * 255).to(torch.uint8).cpu()
        # img= Image.fromarray(image_tensor.permute(1, 2, 0).numpy())
        # img.save(os.path.join(cur_path, 'decoded_image_12.png'))
        image = image[0, -1, 0]  # (c, f, h, w)
        #print(image.shape)  # torch.Size([1, 3, 1024, 1024])
        image = image.cpu().float().unsqueeze(0).permute(0, 2, 3, 1)
        #print(image.shape)  # torch.Size([1, 1024, 1024, 3])

    else:
        mean = torch.tensor(joy_ai_mean, dtype=latents.dtype, device=latents.device)
        std = torch.tensor(joy_ai_std, dtype=latents.dtype, device=latents.device)
        scale = [mean, 1.0 / std]
        latents = latents / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
        #latents=map_neg1_1_to_0_1(latents)
        image=vae.decode(latents) ##Decoded image shape: torch.Size([2, 1, 1024, 1024, 3]), dtype: torch.float32, device: cpu
        image= rearrange(image, "f b h w c -> (f b) h w c")
    #print(f"Decoded image shape: {image.shape}, dtype: {image.dtype}, device: {image.device}") 
    return image



def prepare_conditions( latents, image=None, last_image=None,vae=None):
    """
    Prepare conditional inputs for video generation.

    Args:
        latents: Generated latent tensor with shape (B, N, C, T, latent_H, latent_W)
        image: First frame condition, shape (B, N, 3, 1, H, W)
        last_image: Last frame condition, shape (B, N, 3, 1, H, W)

    Returns:
        Combined condition tensor with shape (B, N, C+1, T, H, W)
    """
    device, dtype = latents.device, latents.dtype
    batch_size, num_items, latent_channels, latent_frames, latent_h, latent_w = latents.shape

    # If no conditions provided, return zero condition
    if image is None and last_image is None:
        return torch.zeros(
            batch_size, num_items, latent_channels + 1, latent_frames, latent_h, latent_w,
            device=device, dtype=dtype
        )

    num_frame = (latent_frames - 1) * 4 + 1
    height = latent_h * 8
    width = latent_w * 8

    # Initialize mask
    mask = torch.zeros(batch_size, num_items, 1, latent_frames,
                        latent_h, latent_w, device=device, dtype=dtype)

    # Build video condition
    if image is not None and last_image is not None:
        # Both first and last frame conditions
        image = image.to(device=device, dtype=dtype)
        last_image = last_image.to(device=device, dtype=dtype)

        middle_frames = torch.zeros(
            batch_size, num_items, image.shape[2], num_frame -
            2, height, width,
            device=device, dtype=dtype
        )
        video_condition = torch.cat(
            [image, middle_frames, last_image], dim=3)
        mask[:, :, :, 0] = 1   # Mark first frame as conditional
        mask[:, :, :, -1] = 1  # Mark last frame as conditional

    elif image is not None:
        # Only first frame condition
        image = image.to(device=device, dtype=dtype)
        remaining_frames = torch.zeros(
            batch_size, num_items, image.shape[2], num_frame -
            1, height, width,
            device=device, dtype=dtype
        )
        video_condition = torch.cat([image, remaining_frames], dim=3)
        mask[:, :, :, 0] = 1   # Mark first frame as conditional
    else:
        raise NotImplementedError

    # VAE encode the video condition
    video_condition = rearrange(
        video_condition, "b n c t h w -> (b n) c t h w")
    latent_condition = vae.encode(
        video_condition).latent_dist.sample()

    # Normalize
    normalize_latents=lambda x: x * (2.0 / x.shape[-1]) # TODO is's not right,just for test

    latent_condition = normalize_latents(latent_condition)

    # Reshape back to (B, N, C, T, H, W)
    latent_condition = rearrange(
        latent_condition, "(b n) c t h w -> b n c t h w", b=batch_size)

    # Concat
    return torch.cat([latent_condition, mask], dim=2)

def prepare_latents_(
        batch_size,
        num_items,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        reference_images=None,
        image=None,
        last_image=None,
        vae= None,
        image_tensor=None

    ):
       
        shape = (
            batch_size,
            num_items,
            num_channels_latents,
            (video_length - 1) // 4 + 1,
            int(height) // 8,
            int(width) // 8,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if reference_images is not None:

                ref_img = [torch.from_numpy(
                    np.array(x.convert("RGB"))) for x in reference_images]
                ref_img = torch.stack(ref_img).to(device=device, dtype=dtype)
                ref_img = ref_img / 127.5 - 1.0
                ref_img = rearrange(ref_img, "x h w c -> x c 1 h w")

                if isinstance(vae, WanxVAE):
                    ref_vae = vae.encode(ref_img) #(torch.Size([1, 16, 1, 128, 128]), torch.float32, True)
                else:
                    mean = torch.tensor(joy_ai_mean, dtype=dtype, device=device)
                    std = torch.tensor(joy_ai_std, dtype=dtype, device=device)
                    scale = [mean, 1.0 / std]
                    ref_vae = vae.encode(image_tensor).to(device=device, dtype=dtype) # (torch.Size([1, 16, 1, 128, 128]), torch.bfloat16, True)
                    ref_vae=map_0_1_to_neg1_1(ref_vae) # comfyUI 0.1 to -1.1
                    ref_vae = (ref_vae - scale[0].view(1, 16, 1, 1, 1)) * scale[1].view(1, 16, 1, 1, 1)

                #print(f"Reference VAE shape: {ref_vae.shape,ref_vae.dtype,ref_vae.is_cuda}")
                ref_vae = rearrange(
                    ref_vae, "(b n) c 1 h w -> b n c 1 h w", n=(num_items - 1))
                #print(f"Reference VAE reshaped: {ref_vae.shape}") # torch.Size([1, 1, 16, 1, 128, 128])
                noise = randn_tensor(
                    (shape[0], 1, *shape[2:]),
                    generator=generator, device=device, dtype=dtype
                )
                latents = torch.cat([ref_vae, noise], dim=1)
            else:
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
        else:
            latents = latents.to(device)
        enable_multi_task=False
        if not enable_multi_task:
            return latents, None

        # image: (b, n, c, 1, h, w), last_image: (b, n, c, 1, h, w)
        condition = prepare_conditions(latents, image, last_image, vae)

        return latents, condition

def get_latents(vae, images, height, width, device,seed,image_tensor, dtype):
    num_items = 1 if images is None or len(
            images) == 0 else 1 + len(images)
    num_channels_latents =16
    num_frames = 1
    generator = torch.Generator(device='cuda').manual_seed(int(seed))
    latents, condition = prepare_latents_(
        1,
        num_items,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        reference_images=images,
        vae=vae,
        image_tensor=image_tensor

    )
    return latents, condition

def load_input_image(image_path: str | None) -> Image.Image | None:
    if not image_path:
        return None
    return Image.open(image_path).convert('RGB')


def is_rank0() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    return torch.device(f'cuda:{local_rank}')


def  load_mmdit(dit_path,gguf_path,offload):

    settings = InferSettings(
        config_path=os.path.join(cur_path, 'infer_config.py') ,
        ckpt_path=dit_path or gguf_path,
        rewrite_model=None ,#'gpt-5'
        openai_api_key=os.environ.get('OPENAI_API_KEY', None),
        openai_base_url=os.environ.get('OPENAI_BASE_URL', None),
        default_seed=42,
        repo_path=os.path.join(cur_path, 'JoyAI-Image-Und'),
        
    )
    
    device = resolve_device() if not  offload else torch.device('cpu')


    model = build_model(
        settings,
        device=device,
        hsdp_shard_dim_override=False,
    )
    return model


def infer_joyai(model,lat,positive,negative, steps, guidance_scale,offload,offload_block_num):

    start_time = time.time()
    output_image = model.infer(
        images=lat.get("images"),
        height=lat["height"],
        width=lat["width"],
        steps=steps,
        guidance_scale=guidance_scale,
        prompt_embeds=positive[0][0] ,
        prompt_embeds_mask=positive[0][1]['prompt_attention_mask'] ,
        negative_prompt_embeds=negative[0][0] ,
        negative_prompt_embeds_mask=negative[0][1]['prompt_attention_mask'] if negative else None,
        offload=offload,
        offload_block_num=offload_block_num,
        lat=lat["samples"]
    )
    elapsed = time.time() - start_time

    print(f'Time taken: {elapsed:.2f} seconds')
    return output_image


