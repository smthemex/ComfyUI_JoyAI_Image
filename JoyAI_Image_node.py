 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from .model_loader_utils import clear_comfyui_cache,save_lat_emb,read_lat_emb,tensor2pillist,phi2narry,tensor2pillist_upscale
from .inference import load_mmdit,infer_joyai,load_vae,get_latents,vae_decode
from .inference_und import get_conditioning,load_qwen3vl_model,encoder_input

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class JoyAI_Image_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAI_Image_SM_Model",
            display_name="JoyAI_Image_SM_Model",
            category="JoyAI_Image",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls,dit,gguf) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        model= load_mmdit(dit_path,gguf_path,True)
        return io.NodeOutput(model)

class JoyAI_Image_SM_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="JoyAI_Image_SM_VAE",
            display_name="JoyAI_Image_SM_VAE",
            category="JoyAI_Image",
            inputs=[
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[io.Vae.Output(display_name="vae"),],
            )
    @classmethod
    def execute(cls,vae ) -> io.NodeOutput:
        clear_comfyui_cache()
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        vae=load_vae(vae_path,device,torch.bfloat16) 
        return io.NodeOutput(vae)
    
class JoyAI_Image_SM_Clip(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="JoyAI_Image_SM_Clip",
            display_name="JoyAI_Image_SM_Clip",
            category="JoyAI_Image",
            inputs=[
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf") ),
            ],
            outputs=[io.Clip.Output(display_name="clip"),],
            )
    @classmethod
    def execute(cls,clip,gguf ) -> io.NodeOutput:
        clear_comfyui_cache()
        safetensors_path=folder_paths.get_full_path("clip", clip) if clip != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        clip=load_qwen3vl_model(safetensors_path,gguf_path)     
        return io.NodeOutput(clip)

class JoyAI_Vae_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="JoyAI_Vae_Decoder",
            display_name="JoyAI_Vae_Decoder",
            category="JoyAI_Image",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latents",),  
            ],
            outputs=
            [io.Image.Output(display_name="image"),],
            )
    @classmethod
    def execute(cls,vae,latents ) -> io.NodeOutput:
        clear_comfyui_cache()
        image=vae_decode(vae,latents)
        return io.NodeOutput(image)


        
class JoyAI_Image_LATENTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAI_Image_LATENTS",
            display_name="JoyAI_Image_LATENTS",
            category="JoyAI_Image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("width", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Vae.Input("vae",optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                ],
            )
    @classmethod
    def execute(cls,image,seed,width,height,vae=None,) -> io.NodeOutput:
        clear_comfyui_cache() 
        # width=(width //32)*32 if width % 32 != 0  else width 
        # height=(height //32)*32 if height % 32 != 0  else height
        images=tensor2pillist_upscale(image,width,height) if image is not None else None
        if vae is None and images is not None:
            raise Exception("When use image,you must provide a vae")
        lat,_=get_latents(vae, images, height, width, device,seed,image, torch.bfloat16) 
        latent={"samples":lat,"width":width,"height":height,"images":images}  
        return io.NodeOutput(latent)

    
class JoyAI_Image_ENCODER(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="JoyAI_Image_ENCODER",
            display_name="JoyAI_Image_ENCODER",
            category="JoyAI_Image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt",multiline=True,default="Turn the plate blue" ),
                io.Combo.Input("infer_device",options= ["cuda","cpu"]  ),
                io.Boolean.Input("save_emb",default=False),
                io.Image.Input("image",optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),

                ],
            )
    @classmethod
    def execute(cls,clip,prompt, infer_device,save_emb,image=None) -> io.NodeOutput:
        clear_comfyui_cache()
        images=tensor2pillist(image) if image is not None else None
        positive,negative=get_conditioning(clip,prompt, images,infer_device)
        if save_emb:
            save_lat_emb("embeds",positive,negative)
        return io.NodeOutput(positive,negative)

class JoyAI_Image_Understand(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="JoyAI_Image_Understand",
            display_name="JoyAI_Image_Understand",
            category="JoyAI_Image",
            inputs=[
                io.Clip.Input("clip"),
                io.Image.Input("image"),
                io.String.Input("prompt",multiline=True,default="Turn the plate blue" ),
                io.Int.Input("max_new_tokens", default=2048, min=256, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("temperature", default=0.7, min=0, max=1,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("top_p", default=0.8, min=0, max=1,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("top_k", default=50, min=1, max=200,step=1,display_mode=io.NumberDisplay.number),
                io.Combo.Input("infer_device",options= ["cuda","cpu"]  ),
            ],
            outputs=[
                io.String.Output(display_name="response"),

                ],
            )
    @classmethod
    def execute(cls,clip,image,prompt,max_new_tokens,temperature,top_p,top_k,infer_device,) -> io.NodeOutput:
        clear_comfyui_cache()
        images=tensor2pillist(image)
        response=encoder_input(clip,prompt,images,max_new_tokens,top_p,top_k,temperature,infer_device)
       
        return io.NodeOutput(response)


class JoyAI_Image_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAI_Image_SM_KSampler",
            display_name="JoyAI_Image_SM_KSampler",
            category="JoyAI_Image",
            inputs=[
                io.Model.Input("model"),     
                io.Latent.Input("latents",),    
                io.Int.Input("steps", default=20, min=1, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=5.0, min=1, max=20,step=0.1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("offload", default=True),
                io.Int.Input("offload_block_num", default=1, min=1, max=40,step=1,display_mode=io.NumberDisplay.number),
                io.Conditioning.Input("positive",optional=True),
                io.Conditioning.Input("negative",optional=True),  
            ], 
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )
    @classmethod
    def execute(cls, model,latents,steps,guidance_scale,offload,offload_block_num,positive=None,negative=None,) -> io.NodeOutput:
        if positive is None:
            positive,negative=read_lat_emb("embeds",device)
        clear_comfyui_cache()
        if not offload:
            model.dit.to(device)
        lat=infer_joyai(model,latents,positive,negative, steps, guidance_scale,offload,offload_block_num)
        if not offload:
            model.dit.to("cpu")
        latent={"samples":lat}  
        return io.NodeOutput(latent)



class JoyAI_Image_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            JoyAI_Image_SM_Model,
            JoyAI_Image_SM_VAE,
            JoyAI_Image_SM_Clip,
            JoyAI_Image_LATENTS,
            JoyAI_Image_SM_KSampler,
            JoyAI_Image_ENCODER,
            JoyAI_Vae_Decoder,
            JoyAI_Image_Understand,
        ]
async def comfy_entrypoint() -> JoyAI_Image_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return JoyAI_Image_SM_Extension()
