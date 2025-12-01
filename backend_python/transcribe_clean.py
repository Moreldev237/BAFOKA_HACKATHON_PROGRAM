# transcribe_clean.py
import os
from typing import Optional, Dict, List
import torch
import whisper as OW  # openai-whisper

class SpeechToText:
    def __init__(self, whisper_model: str = "small", device: Optional[str] = None, callback=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ow_model = None
        self.whisper_model_name = whisper_model
        self.callback = callback if callable(callback) else (lambda p: None)

    def _cb(self, p: float):
        p = max(0.0, min(100.0, float(p)))
        try:
            self.callback(p)
        except:
            pass

    def _load_openai_whisper(self):
        if self._ow_model is None:
            self._ow_model = OW.load_model(self.whisper_model_name, device=self.device)
        return self._ow_model

    def transcribe(self, file_path: str) -> Dict:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        model = self._load_openai_whisper()
        self._cb(5.0)
        audio = OW.load_audio(file_path)
        total_len = len(audio)
        sr = 16000.0
        segments = []

        # traitement en un seul chunk pour simplifier
        result = model.transcribe(audio, verbose=False)
        for seg in result.get("segments", []):
            segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip()
            })

        self._cb(100.0)
        return {"backend": "openai-whisper", "segments": segments}
