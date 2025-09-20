import torch
import gradio as gr
import mlflow.pytorch
from transformers import T5Tokenizer

# ---------------- CONFIG ---------------- #
mt_pretrained_model_name = "csebuetnlp/banglat5_nmt_en_bn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# ---------------- LOAD TOKENIZER ---------------- #
tokenizer = T5Tokenizer.from_pretrained(mt_pretrained_model_name)

# ---------------- LOAD MODEL FROM MLFLOW ---------------- #
RUN_ID = "f5ec797e21654b46a383cfc582813440"
logged_model_uri = f"runs:/{RUN_ID}/mt_model"
print("Loading model from MLflow:", logged_model_uri)

model = mlflow.pytorch.load_model(logged_model_uri)
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
        output_tokens = model.model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ---------------- GRADIO INTERFACE ---------------- #
gr_interface = gr.Interface(
    fn=translate_english_to_bangla,
    inputs=gr.Textbox(lines=3, placeholder="Enter English sentence here...", label="English Text"),
    outputs=gr.Textbox(label="Bangla Translation"),
    title="English → Bangla Translator (from MLflow)",
    description="Translates English into Bangla using model loaded directly from MLflow."
)

# ---------------- LAUNCH ---------------- #
gr_interface.launch(share=True)
