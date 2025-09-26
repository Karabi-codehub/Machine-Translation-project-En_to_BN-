import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIG ---------------- #
mt_pretrained_model_name = "csebuetnlp/banglat5_nmt_en_bn"  # base architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# ---------------- LOAD TOKENIZER ---------------- #
tokenizer = AutoTokenizer.from_pretrained(mt_pretrained_model_name)

# ---------------- LOAD MODEL + YOUR WEIGHTS ---------------- #
# Load the base pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(mt_pretrained_model_name)

# Load your fine-tuned weights (must be in the same folder as app.py)
state_dict = torch.load("mt_model_weights.pt", map_location=device)
model.load_state_dict(state_dict, strict=False)  # strict=False = ignore extra keys
model.to(device)
model.eval()

# ---------------- TRANSLATION FUNCTION ---------------- #
def translate_english_to_bangla(sentence: str) -> str:
    input_ids = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    ).input_ids.to(device)

    with torch.no_grad():
        output_tokens = model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ---------------- GRADIO INTERFACE ---------------- #
gr.Interface(
    fn=translate_english_to_bangla,
    inputs=gr.Textbox(lines=3, placeholder="Enter English sentence here...", label="English Text"),
    outputs=gr.Textbox(label="Bangla Translation"),
    title="English → Bangla Translator (Fine-tuned)",
    description="Translates English into Bangla using your fine-tuned model weights."
).launch()

