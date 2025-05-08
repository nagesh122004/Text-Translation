from flask import Flask, request, render_template, jsonify
from transformers import MarianMTModel, MarianTokenizer
import os

app = Flask(__name__)

# Define language pairs
LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "hi": "Hindi",
    "te": "Telugu"
}

# Model cache
model_cache = {}

# Helper to get Hugging Face model name
def get_model_name(src_lang, tgt_lang):
    return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

# Load model only once and store in cache
def load_model(src_lang, tgt_lang):
    key = f"{src_lang}-{tgt_lang}"
    if key not in model_cache:
        model_name = get_model_name(src_lang, tgt_lang)
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model_cache[key] = (tokenizer, model)
            print(f"Model loaded: {model_name}")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            raise
    return model_cache[key]

@app.route("/")
def home():
    return render_template("index.html", languages=LANGUAGES)

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    src_lang = data.get("src_lang")
    tgt_lang = data.get("tgt_lang")
    text = data.get("text")

    if not src_lang or not tgt_lang:
        return jsonify({"error": "Source or target language not provided"}), 400

    if src_lang == tgt_lang:
        return jsonify({"translated_text": text})

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        tokenizer, model = load_model(src_lang, tgt_lang)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
