# app.py
from flask import Flask, request, jsonify
import os
import time
from werkzeug.utils import secure_filename
import requests

from transcribe_clean import SpeechToText
from openrouter_service import detect_intent_openrouter

# -------------------- Config --------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Whisper --------------------
def progress_callback(p):
    print(f"Progress: {p:.1f}%")

stt = SpeechToText(whisper_model="small", callback=progress_callback)

# -------------------- Endpoint --------------------
@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    filepath = None

    try:
        # 🔹 Cas 1 : fichier uploadé
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "Empty filename"}), 400
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

        # 🔹 Cas 2 : URL audio JSON
        elif request.is_json and "audio_url" in request.json:
            audio_url = request.json["audio_url"]
            ext = os.path.splitext(audio_url)[-1] or ".ogg"
            filename = f"downloaded_{int(time.time())}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            r = requests.get(audio_url)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)

        else:
            return jsonify({"error": "No file or audio_url provided"}), 400

        # -------------------- Transcription --------------------
        result = stt.transcribe(filepath)
        full_text = " ".join([seg["text"] for seg in result.get("segments", [])]).strip()

        if not full_text:
            return jsonify({"text": "", "language": None, "intent": "autre"}), 200

        # -------------------- Langue détectée par Whisper --------------------
        detected_lang = result.get("language", "fr")  # fallback fr si absent

        # -------------------- Détection d’intention --------------------
        intent = detect_intent_openrouter(full_text, detected_lang)

        return jsonify({
            "text": full_text,
            "language": detected_lang,
            "intent": intent
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

# -------------------- Lancement --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
