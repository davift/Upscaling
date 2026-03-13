import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

import config

import gradio as gr
import gradio_client.utils as _gc_utils
import torch
import gc
from datetime import datetime
from PIL import Image
import upscaling

# Monkey-patch: fix gradio_client crash when additionalProperties is a bool
_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = upscaling.pipeline(config.MODEL_INDEX, config.MODEL_SUBINDEX)

def infer(input_image, guidance_scale, steps, progress=gr.Progress(track_tqdm=True)):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    init_image = Image.fromarray(input_image).convert("RGB")

    seed = torch.seed()
    generator = torch.Generator(device=device).manual_seed(seed)

    try:
        image = pipe(
            # Hardcoded prompts
            prompt="high quality, detailed, sharp focus, best quality",
            negative_prompt="ugly, deformed, disfigured, poor quality, low resolution, low quality, blurry",
            image=init_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image.format = "PNG"

        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{seed}.png"
        image.save(os.path.join(config.FULL_PATH, filename))

        return image, seed

    finally:
        gc.collect()
        torch.cuda.empty_cache()

css = """
.progress-view {
    opacity: 0.2 !important;
    background: rgba(255, 255, 255, 0.2) !important;
}
/* This prevents the "ghosting" or clearing of the image container during load */
.generating {
    visibility: visible !important;
}
"""

print(f"Loading web-ui with model index: {config.MODEL_INDEX},{config.MODEL_SUBINDEX}")
with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Upscaling | by DFT.WIKI") as demo:
    gr.Markdown("## 🖌️ Uncensored Upscaling (NSFW) | Use with resposibility and respect!")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label=f"Input Image", type="numpy")
            
        with gr.Column():
            guidance_scale_in = gr.Slider(1, 10, 7, step=0.5, label="Guidance")
            steps_in = gr.Slider(1, 25, 20, step=1, label="Steps")
            out_seed = gr.Number(label="Seed Used")

    run_btn = gr.Button("Upscale", variant="primary")
    out_img = gr.Image(label="Result", format="png")

    run_btn.click(
        fn=infer,
        inputs=[img_in, guidance_scale_in, steps_in],
        outputs=[out_img, out_seed],
        show_progress="minimal"
    )

    with gr.Row():
        bottom_img1 = gr.Image(value="output1.png", show_label=False)
        bottom_img2 = gr.Image(value="output2.png", show_label=False)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    favicon_path="icon.png"
)
