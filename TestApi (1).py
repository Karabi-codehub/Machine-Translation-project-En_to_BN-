import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MT_PRETRAINED_MODEL_NAME = "csebuetnlp/banglat5_nmt_en_bn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MT_PRETRAINED_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MT_PRETRAINED_MODEL_NAME)
state_dict = torch.load("mt_model_weights.pt", map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

def call_translate_api(text: str) -> dict:
    try:
        input_ids = tokenizer(text, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=MAX_LENGTH).input_ids.to(DEVICE)
        with torch.no_grad():
            output_tokens = model.generate(input_ids, max_length=MAX_LENGTH,
                                           num_beams=4, early_stopping=True)
        translation = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return {"status": "success", "translation": translation, "raw_response": output_tokens.tolist()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("===== English → Bangla Translator =====")
    while True:
        text = input("\nEnter English text (or type 'exit' to quit): ")
        if text.strip().lower() == "exit":
            print("Exiting translator. Goodbye!")
            break

        result = call_translate_api(text)
        if result["status"] == "success":
            print(f"Bangla Translation: {result['translation']}")
        else:
            print(f"Error: {result['error']}")
