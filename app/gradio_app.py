import gradio as gr
from utils.history import load_history
from src.engine import TextToImageEngine
from configs.config import MODELS, DEFAULT_MODEL
CUSTOM_CSS = """
.gradio-container {
    max-width: 1600px !important;
    margin: auto;
}

footer {
    display: none !important;
}

h1 {
    text-align: center;
}

button {
    font-size: 17px !important;
    font-weight: bold !important;
    border-radius: 12px !important;
}

.gradio-gallery {
    border-radius: 12px;
}

"""

# ==========================================================
# GLOBAL ENGINE
# ==========================================================

engine = None


def get_engine(model_name=None):
    global engine

    if engine is None:
        engine = TextToImageEngine()

    if model_name is not None:
        engine.load_model(model_name)

    return engine

# ==========================================================
# GENERATE
# ==========================================================

def generate(
    model_name,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    seed,
    num_images,
):

    result = get_engine(model_name).generate(
        model_name=model_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        num_images=num_images,
    )

    gallery = [item["image_path"] for item in result]

    first = result[0]

    info = f"""
# 📝 Generation Details

| Property | Value |
|----------|-------|
| **Prompt** | {first["prompt"]} |
| **Negative Prompt** | {first["negative_prompt"]} |
| **Resolution** | {first["width"]} × {first["height"]} |
| **Inference Steps** | {first["steps"]} |
| **CFG Scale** | {first["cfg"]} |
| **Seed** | {first["seed"]} |
| **Generation Time** | {first["generation_time"]} sec |
"""

    from utils.helper import create_zip

    if len(gallery) == 1:
        download_file = gallery[0]
    else:
        download_file = create_zip(gallery)

    return (
    f"## ✅ Generation Complete ({len(gallery)} image(s))",
    f"### Images Generated : {len(gallery)}",
    gallery,
    info,
    download_file,
    get_history(),
    )

# ==========================================================
# CLEAR
# ==========================================================

def clear():

    return (
        "",
        "",
        "## 🟢 Ready",
        [],
        "## 📋 Generation Details\n\nNothing generated yet.",
        None,
        get_history(),
    )
# HISTORY#
def get_history():

    history = load_history()

    if len(history) == 0:
        return "## 📚 History\n\nNo images generated yet."

    markdown = "## 📚 Recent History\n\n"

    for item in history:

        markdown += (
            f"### 🖼 {item['image']}\n"
            f"- **Prompt:** {item['prompt']}\n"
            f"- **Resolution:** {item['width']} × {item['height']}\n"
            f"- **Steps:** {item['steps']}\n"
            f"- **CFG:** {item['cfg']}\n"
            f"- **Seed:** {item['seed']}\n"
            f"- **Generated:** {item['generated_at']}\n\n"
        )

    return markdown

# ==========================================================
# UI
# ==========================================================

with gr.Blocks(
    title="AI Text-to-Image Generator",  
) as demo:

    gr.Markdown(
        """
# 🖼 AI Text-to-Image Generator
### Stable Diffusion 1.5 • Diffusers • PyTorch

Generate high-quality AI images from natural language prompts.
"""
    )
    with gr.Row():

        # ==========================================================
        # LEFT PANEL
        # ==========================================================

        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL,
                label="🤖 AI Model",
            )
            prompt = gr.Textbox(
                label="📝 Prompt",
                placeholder="Describe the image you want...",
                lines=4,
            )

            negative_prompt = gr.Textbox(
                label="🚫 Negative Prompt",
                placeholder="Things you don't want...",
                lines=2,
            )
            with gr.Accordion("⚙ Advanced Settings", open=False):

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
                    label="Seed (Leave Blank = Random)",
                )

                num_images = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    label="Number of Images",
                )

            with gr.Row():

                generate_btn = gr.Button(
                    value="🚀 Generate",
                    variant="primary",
                    scale=4,
                )

                clear_btn = gr.Button(
                    "🗑 Clear",
                    variant="secondary",
                    scale=1,
                )

        # ==========================================================
        # RIGHT PANEL
        # ==========================================================

        with gr.Column(scale=1):
            image_counter = gr.Markdown(
            "### Images Generated : 0"
            )
            status = gr.Markdown(
            "### Images Generated : 0"
            "## 🟢 Ready",
            )
            output_gallery = gr.Gallery(
                label="🖼 Generated Images",
                columns=2,
                rows=2,
                height=700,
                object_fit="contain",
                preview=True,
            )

            download_output = gr.File(
                label="⬇ Download",
            )

            output_info = gr.Markdown(
                height=260,
            )
                
            """
## 📋 Generation Details
Nothing generated yet.
"""         
            
            history_panel = gr.Markdown(
    value=get_history(),
)
    # ==========================================================
    # GENERATE BUTTON
    # ==========================================================

    generate_btn.click(
        fn=generate,
        inputs=[
            model_selector,
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
            status,
            image_counter,
            output_gallery,
            output_info,
            download_output,
            history_panel,
        ],
        show_progress=True,
    )

    # ==========================================================
    # CLEAR BUTTON
    # ==========================================================

    clear_btn.click(
        fn=clear,
        outputs=[
            prompt,
            negative_prompt,
            status,
            output_gallery,
            output_info,
            download_output,
            history_panel,
        ],
    )

    # ==========================================================
    # FOOTER
    # ==========================================================

    gr.Markdown(
        """
---

<center>

### 🤖 AI Text-to-Image Generator

Built with ❤️ using

**Stable Diffusion 1.5 • Diffusers • PyTorch • Gradio**

</center>

"""
    )

# ==========================================================
# LAUNCH
# ==========================================================

demo.launch(
    share=True,
    debug=True,
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(),
)