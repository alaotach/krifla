from flask import Flask, request, jsonify, send_file
import torch
from TTS.api import TTS
import os

# Initialize Flask app
app = Flask(__name__)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model
print("Loading TTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@app.route("/tts", methods=["POST"])
def generate_tts():
    try:
        # Get JSON payload
        data = request.json
        text = data.get("text")
        speaker_wav = data.get("speaker_wav")
        language = data.get("language", "en")

        if not text or not speaker_wav:
            return jsonify({"error": "Missing required fields 'text' or 'speaker_wav'"}), 400

        # Generate speech
        output_path = "output.wav"
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_path)
        
        # Serve the file
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # Trigger deploy.sh script on push event
        os.system("bash /path/to/deploy.sh")
        return jsonify({"status": "Deployment triggered"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
