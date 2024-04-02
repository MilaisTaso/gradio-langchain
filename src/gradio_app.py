import gradio as gr
from langchain.globals import set_verbose

from core import __set_base_path__
from src.agent.stable_diffusion import generate_sd_prompt


def generate_image(
    prompt: str, models: str, temperature: int, width: float, height: float
):
    response, token_info, image = generate_sd_prompt(
        prompt, models, temperature, width, height
    )

    return (response, image, token_info.total_tokens, token_info.total_cost)


prompt_input = gr.Textbox(label="Prompt", placeholder="Here is Prompt")

model_selector = gr.Radio(
    choices=["gpt-3.5-turbo", "gpt-4"],
    label="Models",
    value="gpt-3.5-turbo",
    type="value",
)

tmp_slider = gr.Slider(minimum=0, maximum=1, step=0.05, label="Temperature")
width_slider = gr.Slider(minimum=512, maximum=2048, step=1, label="Width")
height_slider = gr.Slider(minimum=512, maximum=2048, step=1, label="Height")

output_sd_prompt = gr.TextArea(label="Generated Prompt")
output_image = gr.Image(label="Output Image")
total_tokens = gr.Textbox(label="Total tokens")
total_cost = gr.Textbox(label="Total Cost (chatGPT)")


demo = gr.Interface(
    fn=generate_image,
    inputs=[prompt_input, model_selector, tmp_slider, width_slider, height_slider],
    outputs=[output_sd_prompt, output_image, total_tokens, total_cost],
)

if __name__ == "__main__":
    set_verbose(True)

    demo.queue()
    demo.launch(server_port=7861)
