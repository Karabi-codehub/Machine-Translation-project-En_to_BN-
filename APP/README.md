---
title: English → Bangla Translator
sdk: docker
emoji: 🌖
colorFrom: blue
colorTo: green
pinned: false
---
# English → Bangla Translator

This FastAPI app translates English text into Bangla using a pretrained Transformer model.

## How to Use

### Health Check
- **GET /**  
  Returns a simple welcome message to verify the API is running.

### Single Translation
- **POST /translate**  
- **Request body example:**
```json
{
  "text": "Hello, how are you?",
  "max_new_tokens": 128,
  "num_beams": 4
}
{
  "translation": "হ্যালো, আপনি কেমন আছেন?"
}
{
  "texts": ["Hello", "How are you?"],
  "max_new_tokens": 128,
  "num_beams": 4
}
