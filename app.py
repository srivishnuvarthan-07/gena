import gradio as gr
from huggingface_hub import InferenceClient

# 1. Setup the Client
client = InferenceClient(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    token="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"
)

def generate_image(prompt):
    print(f"Generating image for: {prompt}")
    try:
        image = client.text_to_image(prompt)
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- CSS FOR BRIGHTNESS ---
# This adds a sunny gradient background and brightens the text boxes
bright_css = """
body, .gradio-container {
    background: linear-gradient(120deg, #f6d365 0%, #fda085 100%) !important;
}
#component-0 { /* The main container */
    background: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
    border-radius: 20px;
    padding: 20px;
}
"""

# 2. Create the Theme
# We choose 'orange' and 'amber' for a very energetic, bright look
bright_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="stone"
)

# 3. Create the App
with gr.Blocks(theme=bright_theme, css=bright_css) as demo:
    gr.Markdown("# 🌞 Bright AI Art Generator")
    gr.Markdown("Enter a description below to generate an image.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your prompt", 
                placeholder="e.g., A colorful parrot in a tropical jungle, 4k, vibrant",
                lines=2
            )
            # Using variant='primary' uses the bright orange defined in the theme
            submit_btn = gr.Button("✨ Generate Art ✨", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(label="Generated Image")

    # 4. Connect Logic
    submit_btn.click(fn=generate_image, inputs=text_input, outputs=image_output)

# 5. Launch
if __name__ == "__main__":
    demo.launch()
