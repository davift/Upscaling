import os
import config
from PIL import Image
import time

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(pipe, input_image, prompt, negative_prompt, num_images, num_inference_steps, guidance_scale):
    timestamp = int(time.time())

    init_image = Image.open(input_image).convert("RGB")

    print("Running...")
    for image_index in range(num_images):
        seed = torch.seed()
        generator = torch.Generator(device=device).manual_seed(seed)

        start_time = time.time()
        image = pipe(
            prompt=prompt,                           # Not supported by "CompVis/ldm-super-resolution-4x-openimages"
            negative_prompt=negative_prompt,         # Not supported by "CompVis/ldm-super-resolution-4x-openimages"
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,           # Not supported by "CompVis/ldm-super-resolution-4x-openimages"
            generator=generator,
        ).images[0]
        end_time = time.time()
        elapsed = int(end_time - start_time)

        filename = f"{timestamp}_{config.MODEL_INDEX},{config.MODEL_SUBINDEX}_{image_index}_{seed}.png"
        image.save(config.FULL_PATH + filename)
        os.chmod(config.FULL_PATH + filename, 0o777)
        del image
        torch.cuda.empty_cache()
        print(f"[{elapsed}s] {config.FULL_PATH}{filename}")

    print("Done!")

