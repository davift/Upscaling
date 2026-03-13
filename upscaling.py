import config
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

def pipeline(index, subindex):
    print(f"Loading pipeline {index},{subindex}...")

    if index == 0:
        models = [
            "stabilityai/stable-diffusion-x4-upscaler",           # 4x. The standard.
        ]
        from diffusers import StableDiffusionUpscalePipeline
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            models[subindex],
            cache_dir=config.MODELS,
            torch_dtype=torch.float16,
            use_safetensors=True,
            # local_files_only=True
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'safety_checker'):
            pipe.safety_checker = None

        return pipe

    if index == 1:
        models = [
            "CompVis/ldm-super-resolution-4x-openimages",         # 4x, no prompt.
        ]
        from diffusers import LDMSuperResolutionPipeline
        pipe = LDMSuperResolutionPipeline.from_pretrained(
            models[subindex],
            cache_dir=config.MODELS,
            # use_safetensors=True,
            # local_files_only=True
        )
        pipe.to("cuda")
        if hasattr(pipe, 'safety_checker'):
            pipe.safety_checker = None

        return pipe

    # if index == 2:
    #     models = [
    #         "stabilityai/sd-x2-latent-upscaler",                  # 2x. Bad quality.
    #     ]
    #     from diffusers import StableDiffusionLatentUpscalePipeline
    #     pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
    #         models[subindex],
    #         cache_dir=config.MODELS,
    #         torch_dtype=torch.float16,
    #         use_safetensors=True,
    #         # local_files_only=True
    #     )
    #     pipe.enable_model_cpu_offload()
    #     pipe.enable_attention_slicing()
    #     if hasattr(pipe, 'safety_checker'):
    #         pipe.safety_checker = None

    #     return pipe

