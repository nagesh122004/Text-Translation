from flask import Flask, request, render_template, jsonify
from transformers import MarianMTModel, MarianTokenizer
import re

app = Flask(__name__)

# Dictionary of language pairs
LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "hi": "Hindi",
    # Add more language pairs here
}

# Helper function to build the model name dynamically
def get_model_name(src_lang, tgt_lang):
    return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

@app.route("/")
def home():
    return render_template("index.html", languages=LANGUAGES)

@app.route("/translate", methods=["POST"])
def translate():
    src_lang = request.json.get("src_lang")
    tgt_lang = request.json.get("tgt_lang")
    text = request.json.get("text")

    if not src_lang or not tgt_lang:
        return jsonify({"error": "Source or target language not provided"}), 400

    if src_lang == tgt_lang:
        return jsonify({"error": "Source and target languages must be different"}), 400

    model_name = get_model_name(src_lang, tgt_lang)
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    # Split long text into smaller chunks (by sentence)
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunk_size = 10  # number of sentences per chunk
    translated_chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated_chunks.append(translated_text)

    final_translation = " ".join(translated_chunks)
    return jsonify({"translated_text": final_translation})

if __name__ == "__main__":
    app.run(debug=True)
