from flask import Flask, request, render_template, jsonify
from transformers import MarianMTModel, MarianTokenizer
import re

app = Flask(__name__)

# Supported languages
LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "hi": "Hindi",
    # Add more languages as needed
}

# Cache for loaded models
model_cache = {}

# Construct model name from source and target languages
def get_model_name(src_lang, tgt_lang):
    return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

# Load and cache the tokenizer and model
def load_model(src_lang, tgt_lang):
    key = f"{src_lang}-{tgt_lang}"
    if key not in model_cache:
        model_name = get_model_name(src_lang, tgt_lang)
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model_cache[key] = (tokenizer, model)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")
    return model_cache[key]

@app.route("/")
def home():
    return render_template("index.html", languages=LANGUAGES)

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    print("Incoming request:", data)
    
    src_lang = data.get("src_lang")
    tgt_lang = data.get("tgt_lang")
    text = data.get("text")

    if not src_lang or not tgt_lang:
        return jsonify({"error": "Source or target language not provided"}), 400

    if src_lang == tgt_lang:
        return jsonify({"error": "Source and target languages must be different"}), 400

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        tokenizer, model = load_model(src_lang, tgt_lang)

        # Split input into manageable chunks
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunk_size = 10
        translated_chunks = []

        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            translated_tokens = model.generate(**inputs)
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            translated_chunks.append(translated_text)

        final_translation = " ".join(translated_chunks)
        return jsonify({"translated_text": final_translation})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
