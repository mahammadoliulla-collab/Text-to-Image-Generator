import gradio as gr

from src.engine import TextToImageEngine

# ==========================================
# Global Engine
# ==========================================

engine = None


def get_engine():
    global engine

    if engine is None:
        engine = TextToImageEngine()

    return engine


# ==========================================
# Generate Function
# ==========================================

def generate(
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    seed,
    num_images,
):

    current_engine = get_engine()

    result = current_engine.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        num_images=num_images,
    )

    first = result[0]

    info = f"""
### ✅ Generation Details

**Prompt:** {first['prompt']}

**Resolution:** {first['width']} × {first['height']}

**Inference Steps:** {first['steps']}

**CFG Scale:** {first['cfg']}

**Seed:** {first['seed']}

**Generation Time:** {first['generation_time']} seconds

**Image Path:** {first['image_path']}
"""

    return first["image_path"], info


# ==========================================
# UI
# ==========================================

with gr.Blocks(title="Text-to-Image Generator") as demo:

    gr.Markdown(
        """
# 🖼️ Text-to-Image Generator

### Stable Diffusion 1.5

Generate high-quality AI images from text prompts.
"""
    )

    with gr.Row():

        # --------------------------
        # LEFT PANEL
        # --------------------------

        with gr.Column(scale=1):

            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                placeholder="Describe the image..."
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                placeholder="Things you don't want..."
            )

            with gr.Row():

                width = gr.Dropdown(
                choices=[512, 768],
                value=512,
                label="Width",
            )

                height = gr.Dropdown(
                choices=[512, 768],
                value=512,
                label="Height",
            )

            steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=25,
                step=1,
                label="Inference Steps",
            )

            cfg = gr.Slider(
                minimum=1,
                maximum=15,
                value=7.5,
                step=0.5,
                label="CFG Scale",
            )

            seed = gr.Textbox(
                value="",
                label="Seed (Leave Blank for Random)",
            )

            num_images = gr.Slider(
                minimum=1,
                maximum=4,
                value=1,
                step=1,
                label="Number of Images",
            )

            generate_btn = gr.Button(
                "🚀 Generate Image",
                variant="primary",
            )

        # --------------------------
        # RIGHT PANEL
        # --------------------------

        with gr.Column(scale=1):

            output_image = gr.Image(
                label="Generated Image",
                type="filepath",
            )

            output_info = gr.Markdown()

    generate_btn.click(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            cfg,
            seed,
            num_images,
        ],
        outputs=[
            output_image,
            output_info,
        ],
    )

demo.launch()