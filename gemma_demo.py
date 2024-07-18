from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from huggingface_hub import login

# Use huggingface-cli login
secret_file = r'D:\Documents\Secrets\huggingface_secret.txt'
try:
    with open(secret_file, 'r') as file:
        secret_text = file.read()
except FileNotFoundError:
    print(f"The file '{secret_file}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

login(secret_text)

model_name = 'google/gemma-1.1-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name)  # google/gemma-2-9b-it
model = AutoModelForCausalLM.from_pretrained(
    model_name
)

input_text = "Write me a poem with rhyming lines about a Dungeons and Dragons adventure."
inputs = tokenizer(input_text, return_tensors="pt")
# https://huggingface.co/docs/transformers/v4.42.0/en/internal/generation_utils#transformers.TextStreamer
# https://huggingface.co/docs/text-generation-inference/conceptual/streaming
# https://www.gradio.app/guides/quickstart
streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(**inputs, streamer=streamer, max_length=2000, do_sample=True, temperature=0.9)

