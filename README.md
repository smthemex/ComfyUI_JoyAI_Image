# ComfyUI_JoyAI_Image

[JoyAI-Image](https://github.com/jd-opensource/JoyAI-Image):Awakening Spatial Intelligence in Unified Multimodal Understanding and Generation



Update
-----
* Need 64RAM+8VRAM 


1.Installation  
-----

* In the ./ComfyUI/custom_nodes directory, run the following:   

```
git clone https://github.com/smthemex/ComfyUI_JoyAI_Image
```

2.requirements  
----

```
pip install -r requirements.txt

```

3.checkpoints 
----

* transformers/vae/clip  [links](https://huggingface.co/jdopensource/JoyAI-Image-Edit) 
* or [aliyun](https://pan.quark.cn/s/e20f511c921c)
* or [hg](https://huggingface.co/smthem/JoyAI-Image-Edit-merge-dit-gguf)

```
├── ComfyUI/models/
|     ├── diffusion_models/
|        ├──joy_image_transformer.safetensors
|     ├── vae/
|        ├──Wan2.1_VAE.pth
|     ├── clips
|        ├──JoyAI-Image-Und-merger_bf16.safetensors
```

4.Example
----

![](https://github.com/smthemex/ComfyUI_JoyAI_Image/blob/main/example_workflows/example.png)
![](https://github.com/smthemex/ComfyUI_JoyAI_Image/blob/main/example_workflows/example2.png)
![](https://github.com/smthemex/ComfyUI_JoyAI_Image/blob/main/example_workflows/example3.png)

5.Citation
----

```
@article{JoyAI2023,}

```



