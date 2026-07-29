import gradio as gr

from src.engine import TextToImageEngine
from src.img2img_engine import Img2ImgEngine

from configs.config import MODELS, DEFAULT_MODEL

from utils.history import load_history
from utils.helper import create_zip


# ==========================================================
# CSS
# ==========================================================

CUSTOM_CSS = """
.gradio-container{
    max-width:1600px !important;
    margin:auto;
}

footer{
    display:none !important;
}

h1{
    text-align:center;
}

button{
    font-size:17px !important;
    font-weight:bold !important;
    border-radius:12px !important;
}

.gradio-gallery{
    border-radius:12px !important;
}
"""


# ==========================================================
# GLOBAL ENGINES
# ==========================================================

txt2img_engine = None
img2img_engine = None


# ==========================================================
# TEXT TO IMAGE ENGINE
# ==========================================================

def get_txt2img_engine(model_name=None):

    global txt2img_engine

    if txt2img_engine is None:
        txt2img_engine = TextToImageEngine()

    if model_name is not None:
        txt2img_engine.load_model(model_name)

    return txt2img_engine


# ==========================================================
# IMAGE TO IMAGE ENGINE
# ==========================================================

def get_img2img_engine():

    global img2img_engine

    if img2img_engine is None:
        img2img_engine = Img2ImgEngine()

    return img2img_engine


# ==========================================================
# HISTORY
# ==========================================================

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
# CLEAR
# ==========================================================

def clear():

    return (
        "",
        "",
        "## 🟢 Ready",
        "### Images Generated : 0",
        [],
        "## 📋 Generation Details\n\nNothing generated yet.",
        None,
        get_history(),
    )
# ==========================================================
# TEXT TO IMAGE GENERATION
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

    result = get_txt2img_engine(model_name).generate(
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
## 📋 Generation Details

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

    if len(gallery) == 1:
        download_file = gallery[0]
    else:
        download_file = create_zip(gallery)

    return (
        "## 🟢 Generation Complete",
        f"### Images Generated : {len(gallery)}",
        gallery,
        info,
        download_file,
        get_history(),
    )


# ==========================================================
# IMAGE TO IMAGE GENERATION
# ==========================================================

def generate_img2img(
    image,
    prompt,
    negative_prompt,
    strength,
    steps,
    cfg,
    seed,
):

    if image is None:

        return (
            None,
            "## ❌ Please upload an image first.",
        )

    result = get_img2img_engine().generate(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        steps=steps,
        cfg=cfg,
        seed=seed,
    )

    info = f"""
## 🎨 Image-to-Image Details

| Property | Value |
|----------|-------|
| **Resolution** | {result["width"]} × {result["height"]} |
| **Seed** | {result["seed"]} |
| **Generation Time** | {result["generation_time"]} sec |
"""

    return (
        result["image_path"],
        info,
    )
# ==========================================================
# UI
# ==========================================================

with gr.Blocks(
    title="AI Image Studio",
) as demo:

    gr.Markdown(
        """
# 🖼 AI Image Studio

### Stable Diffusion • Image-to-Image

Generate and edit AI images
"""
    )

    with gr.Tabs():

        # ==========================================================
        # TEXT TO IMAGE
        # ==========================================================

        with gr.Tab("🖼 Text to Image"):

            with gr.Row():

                # ======================================
                # LEFT PANEL
                # ======================================

                with gr.Column(scale=1):

                    model_selector = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value=DEFAULT_MODEL,
                        label="🤖 AI Model",
                    )

                    prompt = gr.Textbox(
                        label="📝 Prompt",
                        placeholder="Describe the image...",
                        lines=4,
                    )

                    negative_prompt = gr.Textbox(
                        label="🚫 Negative Prompt",
                        placeholder="Things you don't want...",
                        lines=2,
                    )

                    with gr.Accordion(
                        "⚙ Advanced Settings",
                        open=False,
                    ):

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
                            label="Seed (Blank = Random)",
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
                            "🚀 Generate",
                            variant="primary",
                            scale=4,
                        )

                        clear_btn = gr.Button(
                            "🗑 Clear",
                            scale=1,
                        )
                # ======================================
                # RIGHT PANEL
                # ======================================

                with gr.Column(scale=1):

                    status = gr.Markdown("## 🟢 Ready")

                    image_counter = gr.Markdown(
                        "### Images Generated : 0"
                    )

                    output_gallery = gr.Gallery(
                        label="🖼 Generated Images",
                        columns=2,
                        rows=2,
                        object_fit="contain",
                        height=600,
                    )

                    download_output = gr.File(
                        label="⬇ Download Image(s)",
                    )

                    output_info = gr.Markdown(
                        "## 📋 Generation Details\n\nNothing generated yet."
                    )

                    history_panel = gr.Markdown(
                        value=get_history()
                    )

        # ==========================================================
        # IMAGE TO IMAGE
        # ==========================================================

        with gr.Tab("🎨 Image to Image"):

            gr.Markdown(
                """
## 🎨 Image-to-Image

Upload an image and transform it using AI.
"""
            )

            with gr.Row():

                with gr.Column(scale=1):

                    input_image = gr.Image(
                        label="📤 Upload Image",
                        type="pil",
                    )

                    img_prompt = gr.Textbox(
                        label="📝 Prompt",
                        lines=3,
                        placeholder="Describe how you want to transform the image...",
                    )

                    img_negative_prompt = gr.Textbox(
                        label="🚫 Negative Prompt",
                        lines=2,
                    )

                    img_strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.75,
                        step=0.05,
                        label="Strength",
                    )

                    img_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Inference Steps",
                    )

                    img_cfg = gr.Slider(
                        minimum=1,
                        maximum=15,
                        value=7.5,
                        step=0.5,
                        label="CFG Scale",
                    )

                    img_seed = gr.Textbox(
                        value="",
                        label="Seed (Blank = Random)",
                    )

                    img_generate_btn = gr.Button(
                        "🚀 Generate",
                        variant="primary",
                    )

                with gr.Column(scale=1):

                    img_output = gr.Image(
                        label="🖼 Generated Image",
                    )

                    img_info = gr.Markdown(
                        "## 📋 Image-to-Image Details"
                    )
    # ==========================================================
    # TEXT TO IMAGE CALLBACKS
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

    clear_btn.click(
        fn=clear,
        outputs=[
            prompt,
            negative_prompt,
            status,
            image_counter,
            output_gallery,
            output_info,
            download_output,
            history_panel,
        ],
    )

    # ==========================================================
    # IMAGE TO IMAGE CALLBACKS
    # ==========================================================

    img_generate_btn.click(
        fn=generate_img2img,
        inputs=[
            input_image,
            img_prompt,
            img_negative_prompt,
            img_strength,
            img_steps,
            img_cfg,
            img_seed,
        ],
        outputs=[
            img_output,
            img_info,
        ],
        show_progress=True,
    )

    # ==========================================================
    # FOOTER
    # ==========================================================

    gr.Markdown(
        """
---

<center>

### 🤖 AI Image Studio

Built with ❤️ using

**Stable Diffusion • Diffusers • PyTorch • Gradio**

</center>
"""
    )


# ==========================================================
# LAUNCH
# ==========================================================

demo.launch(
    share=True,
    debug=True,
)