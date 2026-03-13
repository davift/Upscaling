#!/usr/bin/env python3

import sys
import config

from dotenv import load_dotenv
load_dotenv()

"""
Usage: INDEX=M,N python app.py input_image [prompt] [num_images]
"""

if len(sys.argv) > 1:
    input_image = sys.argv[1]

    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    else:
        prompt = "high quality, detailed, sharp focus, best quality"

    negative_prompt = "ugly, deformed, disfigured, poor quality, low resolution, low quality, blurry"

    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if num_images < 1:
        num_images = 1000000000
        print("Generating images until interrupted...")

    num_inference_steps = 20
    guidance_scale = 7
else:
    input_image = None

if __name__ == "__main__":
    try:
        if input_image:
            print('Entering cli mode...')
            import upscaling
            pipe = upscaling.pipeline(config.MODEL_INDEX, config.MODEL_SUBINDEX)
            import cli
            cli.main(pipe, input_image, prompt, negative_prompt, num_images, num_inference_steps, guidance_scale)
        else:
            import web
            print('Entering web mode...')

    except KeyboardInterrupt:
        print("\nScript interrupted!")

